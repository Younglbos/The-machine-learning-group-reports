"""Run the teacher's full-data, missing-view and sample-stream baselines."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from data.load_dataset import load_dataset
from data.missing import make_mask, mask_summary
from data.preprocessing import dense_views, validate_views, prepare_observed, FrozenStreamEncoder
from evaluation.metrics import scores, silhouette, clustering_accuracy
from evaluation.records import Measurement, aggregate, environment, logger_at, write_csv, write_json, file_hash
from models.kmeans import fit_kmeans
from models.stream_baselines import StreamKMeans

ROOT = Path(__file__).resolve().parents[1]


def load(name, selected=None):
    views, labels = load_dataset(name)
    views = dense_views(views)
    validate_views(views, labels)
    if selected is not None:
        views = [views[index] for index in selected]
    return views, labels


def trial(features, labels, seed, config, info, path):
    # Evaluation, silhouette and file writes are excluded from fit/predict timing.
    with Measurement() as timing:
        model, predicted = fit_kmeans(features, len(np.unique(labels)), seed, config["kmeans"])
    row = {**info, "seed": seed, **scores(labels, predicted), "fit_predict_s": timing.seconds,
           "peak_rss_mb": timing.peak_rss_mb, "iterations": int(model.n_iter_),
           "silhouette": silhouette(features, predicted, seed, config["silhouette_sample_size"])}
    np.savez_compressed(path, predicted=predicted, truth=labels, sample_indices=np.arange(len(labels)))
    return row


def complete(config, out, log):
    rows = []
    for name in config["complete_datasets"]:
        views, labels = load(name)
        start = time.perf_counter()
        blocks, _ = prepare_observed(views, np.ones((len(labels), len(views)), dtype=bool))
        preprocess = time.perf_counter() - start
        methods = [("concat_complete", np.concatenate(blocks, axis=1))]
        methods += [(f"single_complete_v{j+1}", view) for j, view in enumerate(blocks)]
        for method, features in methods:
            for seed in config["seeds"]:
                info = {"dataset": name, "method": method, "samples": len(labels),
                        "views": len(views), "features": features.shape[1], "preprocess_s": preprocess}
                row = trial(features, labels, seed, config, info, out / "labels" / f"complete_{name}_{method}_{seed}.npz")
                rows.append(row)
                log.info("complete %s %s seed=%d ACC=%.4f NMI=%.4f ARI=%.4f", name, method, seed, row['acc'], row['nmi'], row['ari'])
                write_csv(out / "complete_runs.csv", rows)
    write_csv(out / "complete_summary.csv", aggregate(rows, ["dataset", "method"],
              ["acc", "nmi", "ari", "silhouette", "fit_predict_s", "peak_rss_mb"]))


def scenario_a(config, out, log):
    setup, rows, masks = config["scenario_a"], [], []
    views, labels = load(setup["dataset"], setup["view_indices"])
    for pattern in setup["patterns"]:
        for rate in setup["missing_rates"]:
            for seed in config["seeds"]:
                mask = make_mask(len(labels), len(views), rate, seed, pattern, labels)
                tag = f"{pattern}_{rate:.2f}_{seed}"
                np.savez_compressed(out / "masks" / f"{tag}.npz", mask=mask, sample_indices=np.arange(len(labels)))
                masks.append({"pattern": pattern, "requested_rate": rate, "seed": seed, **mask_summary(mask)})
                for fill in setup["fill_methods"]:
                    start = time.perf_counter()
                    blocks, _ = prepare_observed(views, mask, fill)
                    features = np.concatenate(blocks, axis=1)
                    preprocess = time.perf_counter() - start
                    info = {"dataset": setup["dataset"], "method": f"{fill}_fill_kmeans", "pattern": pattern,
                            "requested_rate": rate, "actual_block_rate": float((~mask).mean()),
                            "actual_incomplete_rate": float((~mask.all(axis=1)).mean()), "preprocess_s": preprocess}
                    row = trial(features, labels, seed, config, info, out / "labels" / f"a_{tag}_{fill}.npz")
                    rows.append(row)
                    log.info("A %s rate=%.2f %s seed=%d ACC=%.4f", pattern, rate, fill, seed, row['acc'])
                    write_csv(out / "scenario_a_runs.csv", rows)
    write_json(out / "mask_summary.json", {"definition": setup["rate_definition"], "masks": masks})
    write_csv(out / "scenario_a_summary.csv", aggregate(rows, ["dataset", "method", "pattern", "requested_rate"],
              ["acc", "nmi", "ari", "silhouette", "fit_predict_s", "preprocess_s", "peak_rss_mb"]))


def scenario_a_reference(config, out, log):
    """Use the exact HOG/GIST view subset of scenario A for complete references."""
    setup, rows = config['scenario_a'], []
    views, labels = load(setup['dataset'], setup['view_indices'])
    start=time.perf_counter()
    blocks,_=prepare_observed(views,np.ones((len(labels),len(views)),dtype=bool))
    preprocess=time.perf_counter()-start
    methods=[('concat_complete_reference',np.concatenate(blocks,axis=1))]
    methods += [(f'single_complete_reference_v{j+1}',view) for j,view in enumerate(blocks)]
    for method,features in methods:
        for seed in config['seeds']:
            info={'dataset':setup['dataset'],'method':method,'views':'HOG+GIST',
                  'preprocess_s':preprocess,'reference_uses_complete_data':True}
            row=trial(features,labels,seed,config,info,out/'labels'/f'a_reference_{method}_{seed}.npz')
            rows.append(row)
            log.info('A complete reference %s seed=%d ACC=%.4f',method,seed,row['acc'])
    write_csv(out/'scenario_a_complete_reference_runs.csv',rows)
    write_csv(out/'scenario_a_complete_reference_summary.csv',aggregate(rows,['dataset','method'],
        ['acc','nmi','ari','silhouette','fit_predict_s','peak_rss_mb']))


def recovery_rows(rows, setup):
    output = []
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['condition'], row['seed'], row['method']), []).append(row)
    for (dataset, condition, seed, method), values in groups.items():
        values = sorted(values, key=lambda r: r['batch'])
        pre = values[setup['drift_batch'] - 2]['acc']
        post = [v for v in values if v['batch'] >= setup['drift_batch']]
        threshold = 0.9 * pre
        below = [v for v in post if v['acc'] < threshold]
        recovery, recovered = None, None
        if condition != 'none':
            if not below:
                recovery, recovered = 0, True
            else:
                recovered_values = [v for v in post if v['batch'] > below[0]['batch'] and v['acc'] >= threshold]
                recovered = bool(recovered_values)
                if recovered:
                    recovery = recovered_values[0]['batch'] - setup['drift_batch']
        output.append({'dataset': dataset, 'condition': condition, 'seed': seed, 'method': method,
            'pre_drift_acc': pre, 'recovery_threshold': threshold, 'drop_from_pre_to_min_post': pre - min(v['acc'] for v in post),
            'recovery_batches': recovery, 'recovered': recovered})
    return output


def scenario_b(config, out, log):
    setup, rows = config["scenario_b"], []
    views, labels = load(setup["dataset"])
    clusters = len(np.unique(labels))
    for seed in config["seeds"]:
        order = np.random.default_rng(seed).permutation(len(labels))
        batches = np.array_split(order, setup["batches"])
        write_json(out / "streams" / f"order_{seed}.json", {'batches': [x.tolist() for x in batches], 'seed': seed})
        for condition in setup["drift_conditions"]:
            methods = {name: StreamKMeans(name, clusters, seed, config['kmeans'], setup['window_batches'])
                       for name in ['warm_start', 'sliding_window', 'full_retrain']}
            cumulative = {name: 0.0 for name in methods}
            previous_probe, first_probe_acc = {}, {}
            rng = np.random.default_rng(seed + 1000)
            scale = views[0][batches[0]].std(axis=0)
            shift = rng.normal(size=views[0].shape[1]).astype(np.float32) * scale * setup['shift_std']
            for batch, indices in enumerate(batches, start=1):
                raw = [view[indices].copy() for view in views]
                if condition != 'none' and batch >= setup['drift_batch']:
                    if condition == 'mean_shift':
                        raw[0] += shift
                    else:
                        raw[0] += rng.normal(size=raw[0].shape).astype(np.float32) * scale * setup['noise_std']
                start = time.perf_counter()
                if batch == 1:
                    encoder = FrozenStreamEncoder(setup['pca_components_per_view'], seed).fit(raw)
                features = encoder.transform(raw)
                preprocessing = time.perf_counter() - start
                if batch == 1:
                    probe_indices = indices[:setup['probe_samples']]
                    probe_features = features[:len(probe_indices)].copy()
                    probe_truth = labels[probe_indices]
                for name, model in methods.items():
                    with Measurement() as timing:
                        predicted = model.update(features)
                    elapsed = preprocessing + timing.seconds
                    cumulative[name] += elapsed
                    probe_pred = model.predict(probe_features)
                    retention = scores(probe_truth, probe_pred)['acc']
                    first_probe_acc.setdefault(name, retention)
                    row = {'dataset': setup['dataset'], 'condition': condition, 'seed': seed, 'batch': batch,
                        'method': name, 'samples': len(indices), **scores(labels[indices], predicted),
                        'preprocess_s': preprocessing, 'update_predict_s': timing.seconds, 'batch_s': elapsed,
                        'cumulative_s': cumulative[name], 'peak_rss_mb': timing.peak_rss_mb,
                        'model_state_mb': model.state_bytes / 1024**2,
                        'budget_s': setup['batch_time_budget_s'], 'budget_exceeded': elapsed > setup['batch_time_budget_s'],
                        'probe_acc': retention, 'forgetting_from_initial': first_probe_acc[name] - retention,
                        'probe_label_change': None if name not in previous_probe else 1 - clustering_accuracy(previous_probe[name], probe_pred)}
                    previous_probe[name] = probe_pred.copy()
                    np.savez_compressed(out / 'labels' / f'b_{condition}_{seed}_{batch}_{name}.npz',
                        sample_indices=indices, predicted=predicted, truth=labels[indices],
                        probe_indices=probe_indices, probe_predicted=probe_pred, probe_truth=probe_truth)
                    rows.append(row)
                log.info('B condition=%s seed=%d batch=%d/%d', condition, seed, batch, setup['batches'])
                write_csv(out / 'scenario_b_runs.csv', rows)
    write_csv(out / 'scenario_b_summary.csv', aggregate(rows, ['dataset','condition','method','batch'],
        ['acc','nmi','ari','batch_s','cumulative_s','peak_rss_mb','model_state_mb','probe_acc','probe_label_change','forgetting_from_initial']))
    write_csv(out / 'drift_recovery.csv', recovery_rows(rows, setup))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baselines.json')
    parser.add_argument('--stage', choices=['complete','a','a_reference','b','all'], default='all')
    args = parser.parse_args()
    path = ROOT / args.config
    config = json.loads(path.read_text(encoding='utf-8'))
    out = ROOT / config['result_dir']
    for folder in ['labels','masks','streams','figures']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    log = logger_at(out)
    write_json(out / 'config.json', config)
    write_json(out / 'run_config.json', config)
    write_json(out / 'provenance.json', {'origin':'direct experiment execution',
        'config_source':path.resolve().relative_to(ROOT).as_posix(),
        'recorded_config':'run_config.json','stage':args.stage})
    write_json(out / 'environment.json', environment())
    write_json(out / 'data_hashes.json', {p.name: file_hash(p) for p in (ROOT/'data/raw').glob('*.mat')})
    log.info('Begin stage=%s; seeds=%s; CPU threads=%d', args.stage, config['seeds'], config['cpu_threads'])
    with threadpool_limits(limits=config['cpu_threads']):
        if args.stage in ['complete','all']:
            complete(config, out, log)
        if args.stage in ['a','all']:
            scenario_a(config, out, log)
        if args.stage in ['a_reference','all']:
            scenario_a_reference(config, out, log)
        if args.stage in ['b','all']:
            scenario_b(config, out, log)
    log.info('Finished stage=%s', args.stage)


if __name__ == '__main__':
    main()

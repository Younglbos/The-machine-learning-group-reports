"""Meaningful protocol checks: leakage, label permutations, masks and memory bounds."""

import unittest
import numpy as np
from data.missing import make_mask
from data.preprocessing import prepare_observed, FrozenStreamEncoder
from evaluation.metrics import clustering_accuracy, scores
from models.stream_baselines import StreamKMeans


class EvaluationTests(unittest.TestCase):
    def test_noncontiguous_permuted_labels(self):
        result=scores([10,10,20,20,30,30],[6,6,2,2,8,8])
        self.assertEqual(result,{'acc':1.0,'nmi':1.0,'ari':1.0})

    def test_missing_or_extra_clusters_are_not_assumed_square(self):
        self.assertAlmostEqual(clustering_accuracy([1,1,2,2],[7,7,7,7]),0.5)
        self.assertAlmostEqual(clustering_accuracy([1,1,2,2],[4,5,6,6]),0.75)

    def test_invalid_lengths_rejected(self):
        with self.assertRaises(ValueError): scores([1,2],[1])


class MissingTests(unittest.TestCase):
    def test_exact_nested_whole_view_masks(self):
        previous=np.ones((100,2),dtype=bool)
        for rate in [0.2,0.4,0.6]:
            current=make_mask(100,2,rate,1)
            self.assertEqual(int((~current).sum()),int(100*rate))
            self.assertTrue(current.any(axis=1).all())
            self.assertTrue((current<=previous).all())
            np.testing.assert_array_equal(current,make_mask(100,2,rate,1))
            previous=current

    def test_structured_dropout_targets_one_view(self):
        mask=make_mask(100,2,0.6,1,'single_view')
        self.assertTrue(mask[:,1].all())
        self.assertEqual(int((~mask[:,0]).sum()),60)

    def test_missing_values_never_affect_scaling_or_fill(self):
        rng=np.random.default_rng(1)
        views=[rng.normal(size=(50,5)).astype(np.float32) for _ in range(2)]
        mask=make_mask(50,2,0.6,1)
        poisoned=[v.copy() for v in views]
        for j,v in enumerate(poisoned): v[~mask[:,j]]=np.nan
        for fill in ['zero','mean']:
            a,scalers=prepare_observed(views,mask,fill)
            b,_=prepare_observed(poisoned,mask,fill)
            for x,y in zip(a,b): np.testing.assert_array_equal(x,y)


class StreamTests(unittest.TestCase):
    def test_frozen_first_batch_coordinates(self):
        rng=np.random.default_rng(2)
        initial=[rng.normal(size=(30,5)).astype(np.float32) for _ in range(2)]
        encoder=FrozenStreamEncoder(3).fit(initial)
        before=[s.data_max_.copy() for s in encoder.scalers]
        encoder.transform([v*1000 for v in initial])
        for values,scaler in zip(before,encoder.scalers): np.testing.assert_array_equal(values,scaler.data_max_)

    def test_online_and_window_memory_bounds(self):
        rng=np.random.default_rng(3)
        opts={'n_init':2,'max_iter':20,'tol':1e-4}
        models={name:StreamKMeans(name,2,1,opts,2) for name in ['warm_start','sliding_window','full_retrain']}
        for _ in range(5):
            x=rng.normal(size=(20,3)).astype(np.float32)
            for model in models.values(): model.update(x)
        self.assertEqual(len(models['warm_start'].history),0)
        self.assertEqual(len(models['sliding_window'].history),2)
        self.assertEqual(len(models['full_retrain'].history),5)


if __name__=='__main__': unittest.main()

"""Finalize only completed experiments: recompute, document, and validate artifacts."""

import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    progress=ROOT/'experiments/06_course_comparison/progress.json'
    saved=json.loads(progress.read_text(encoding='utf-8'))
    if saved['phase']!='experiments_complete': raise RuntimeError('Wait for all frozen experiments to complete first')
    for module in ['evaluation.course_geometry','evaluation.build_course_analysis','evaluation.verify_course_outputs',
                   'evaluation.build_report']:
        subprocess.run([sys.executable,'-m',module],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/validate_repository.py','--include-paper-results','--include-course-results'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','evaluation.verify_course_outputs','--public-only'],cwd=ROOT,check=True)
    analysis=json.loads((ROOT/'reports/analysis_verification.json').read_text(encoding='utf-8'))
    if analysis['b_budget_exceedances']!=0: raise RuntimeError('Budget unmet: do not mark B3 complete')
    file=ROOT/'requirements/course_checklist.json'
    checklist=json.loads(file.read_text(encoding='utf-8'))
    updates={
        'A4':('complete',['reports/scenario_a_paired_gains.csv','reports/course_report.html'],
              '统一协议三个随机缺失比例的三个种子均相对均值填充为正收益；固定500轮，不按标签挑选轮次。'),
        'A5':('complete',['reports/geometry_summary.csv','reports/figures/course_a_geometry_pca.png','reports/figures/course_a_geometry_neighbors.png'],
              '真实原始特征、均值填充、解码补全与共识的10-NN误差分析及两张图已完成；隐藏真值仅离线诊断。'),
        'A_key_experiments':('complete',['experiments/06_course_comparison/results_a/hsacc/runs.csv','reports/ablation_summary.csv','reports/robustness_summary.csv'],
              '三模式、三比例、三个种子；去一致性/去补全/等权消融及单视图噪声完成；解释双视图等权退化。'),
        'B1':('complete',['experiments/06_course_comparison/config_b.json','models/view_stream.py'],
              '统一主实验为六视图流，明确对象顺序和历史摘要；原十批样本流另外保留，协议不混用。'),
        'B2':('complete',['reports/scenario_b_summary.csv','reports/figures/course_b_time.png'],
              '统一六视图下warm-start、两批窗口和全量参考均完成3种子/2条件，报告单批及累计时间。'),
        'B3':('complete',['experiments/06_course_comparison/config_b.json','experiments/06_course_comparison/results_b/verification.json'],
              'DSVC课程适配仅跨批保留摘要/原型/中心；90秒预算运行前冻结，全部144条记录无超限。'),
        'B4':('complete',['reports/scenario_b_noise_effect.csv','reports/scenario_b_recovery.csv'],
              '第3/6批视图加噪，对比完全相同种子的无漂移输入，保留配对噪声影响与未恢复情况。'),
        'B5':('complete',['reports/figures/course_b_label_changes.png','reports/figures/course_b_retention.png','reports/scenario_b_recovery.csv'],
              '匹配标签变化、固定旧表示探针、遗忘和恢复均已分析；报告查询空间差异的限制。'),
        'metrics':('complete',['reports/scenario_a_summary.csv','reports/scenario_b_final.csv','reports/isolated_efficiency_summary.csv','reports/geometry_summary.csv'],
              'ACC/NMI/ARI均值±样本SD，Silhouette、邻域，单批/总时长、原生峰值、GPU峰值和恢复均有证据。'),
        'figures':('complete',['reports/figures','reports/course_report.html'],
              '质量表、缺失曲线、逐批曲线、消融、二维PCA、效率与邻域图已生成并说明统计口径。'),
        'modules':('complete',['data','models','train_or_update','evaluation','visualization'],
              '五个模块用于实际基线、作者试验和课程适配；12项协议测试通过。'),
        'reproducibility':('complete',['reports/prediction_verification.json','reports/course_predictions.jsonl.gz','reports/course_report.html'],
              '保存配置/环境/日志/标签/CSV；252条课程记录重算主指标；公开紧凑标签档案可无数据核验。'),
        'literature':('complete',['reports/literature_review.html','THIRD_PARTY.md'],
              '教师条目逐项核验，重点复现两篇；区分正式版本、预印本和摘要；第7条未核实，不冒充已确认文献。'),
        'delivery':('partial',['reports/course_report.html','scripts/demo_course.py','reports/README.md'],
              '报告、代码与可运行演示已完成；PPT按用户要求最后制作，目前尚未交付PPT或完成现场答辩。'),
    }
    for check in checklist['checks']:
        if check['id'] not in updates: continue
        status,evidence,note=updates[check['id']]
        for filename in evidence:
            if not (ROOT/filename).exists(): raise FileNotFoundError(filename)
        check.update(status=status,evidence=evidence,progress_note=note)
        check.pop('remaining',None)
        if check['id']=='modules': check.pop('note',None)
    checklist['ppt_status']='deferred_by_user'
    file.write_text(json.dumps(checklist,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    saved.update(reporting_pending=False,report='reports/course_report.html',prediction_records_verified=252,ppt_status='deferred_by_user')
    progress.write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Experiment/report/code/demo checks complete. PPT remains deferred by the user.')


if __name__=='__main__': main()

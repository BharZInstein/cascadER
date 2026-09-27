"""Evaluate a frozen model on untouched holdout reference groups."""
import argparse,json
from pathlib import Path
import numpy as np
from pipeline import normalize,log
from group_policy import apply_policies

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True);p.add_argument('--model',required=True);a=p.parse_args()
    cache,model_dir=Path(a.cache),Path(a.model);meta=json.loads((cache/'metadata.json').read_text());config=json.loads((model_dir/'model_config.json').read_text())
    assert meta['feature_names']==config['feature_names'];anchors=meta['anchors'];n=len(anchors);m=meta['pairs']
    group=np.memmap(cache/'groups.i4',dtype=np.int32,mode='r',shape=(m,));y=np.memmap(cache/'labels.i1',dtype=np.int8,mode='r',shape=(m,));targets=np.memmap(cache/'targets.s32',dtype='S32',mode='r',shape=(m,))
    x=np.memmap(cache/'features.f32',dtype=np.float32,mode='r',shape=(m,len(meta['feature_names'])))
    held=np.array([r['split']>=8 for r in anchors]);pairs=held[group];countries=np.array([normalize(r['country']) for r in anchors]);actual=np.array([r['true_count'] for r in anchors])
    if config['model_kind']=='lightgbm':
        from lightgbm import Booster
        model=Booster(model_file=str(model_dir/config['model_file']));scores=np.asarray(model.predict(x[pairs],num_threads=4),dtype=np.float32);importance=model.feature_importance(importance_type='gain')
    else:
        from xgboost import XGBClassifier
        model=XGBClassifier();model.load_model(model_dir/config.get('model_file','model.ubj'));model.set_params(n_jobs=4);scores=model.predict_proba(x[pairs])[:,1];importance=model.feature_importances_
    groups=group[pairs];labels=y[pairs];chosen=apply_policies(scores,groups,countries,config,targets[pairs])
    counts=np.bincount(groups[chosen],minlength=n);hits=np.bincount(groups[chosen],weights=labels[chosen],minlength=n)
    den=.25*actual+counts;f=np.divide(1.25*hits,den,out=np.zeros(n),where=den>0);f[actual==0]=counts[actual==0]==0
    values=f[held];rng=np.random.default_rng(42);boot=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(1000)])
    report=json.loads((model_dir/'metrics.json').read_text())
    report.update(holdout_macro_f05=float(values.mean()),holdout_link_precision=float(labels[chosen].sum()/max(1,chosen.sum())),
        holdout_link_recall=float(labels[chosen].sum()/actual[held].sum()),holdout_singletons=int((held&(actual==0)).sum()),
        holdout_singleton_accuracy=float(f[held&(actual==0)].mean()),
        holdout_by_country={c:float(f[held&np.array([r['country']==c for r in anchors])].mean()) for c in sorted({r['country'] for r in anchors})},
        bootstrap_95_percent_interval=list(map(float,np.quantile(boot,[.025,.975]))),
        uncertainty_scope='Sampling uncertainty within these held-out groups only; excludes distribution shifts.',
        feature_importance=dict(zip(meta['feature_names'],map(float,importance))),
        validation_scope='Reserved groups with split >= 8 in the supplied cache. Unique assignment evaluated within those groups.')
    (model_dir/'metrics.json').write_text(json.dumps(report,indent=2));np.save(model_dir/'holdout_scores.npy',scores)
    np.save(model_dir/'holdout_group_scores.npy',values);log(json.dumps(report,indent=2))

if __name__=='__main__':main()

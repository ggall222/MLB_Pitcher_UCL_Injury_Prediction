"""Focused interpretation coverage for profile and workload/departure hypotheses.

These diagnostics do not choose or revise any model. They use already-frozen
validation selections. Profile-only pairs are explicitly labeled supplemental
interpretation probes; no transfer outcomes choose their cutoffs or shapes.
"""
import json
import numpy as np
import pandas as pd
from .ucl_nonlinear_common import OUT,CACHE,CFG,inputs,table,write_json,checkpoint,valid_checkpoint
from .ucl_nonlinear_models import fit,load_model
from .ucl_nonlinear_effects import effect_tables

def run_probes():
    if valid_checkpoint('probes'):return
    f,e,s=inputs();train=f.loc[f.partition.eq('train')&f.target.notna()]
    freeze=json.loads((OUT/'frozen_protocol.json').read_text());reference=pd.read_parquet(CACHE/'training_effect_reference.parquet')
    ids=list(dict.fromkeys([freeze['ranking_leaders']['Boosting'],freeze['per_representation']['Profile + departure|Boosting'],freeze['per_representation']['Previous 62|Boosting']]))
    extra=[['avg_fb_velo_baseline_mean','extension_avg_baseline_mean'],['release_z_sd_baseline_mean','age_at_appearance']]
    plan={}
    for mid in ids:
        features=freeze['catalog'][mid]['features']
        plan[mid]=dict(features=[c for c in CFG['interpretation_features'] if c in features],pairs=[p for p in CFG['interaction_pairs']+extra if set(p)<=set(features)])
    write_json(dict(models=plan,purpose='Interpretation coverage only; no new performance selection or transfer evaluation',profile_only_pairs='supplemental local diagnostics, not prevalidated physiological hypotheses'),OUT/'mechanism_probe_plan.json')
    rows=[];inter=[];groups={p:g for p,g in train.groupby('pitcher_id')};people=np.array(list(groups));rng=np.random.default_rng(CFG['seed'])
    samples=[None]+[rng.choice(people,len(people),replace=True) for _ in range(CFG['training_refits'])]
    for i,draw in enumerate(samples):
        rep=i-1;sample=None if draw is None else pd.concat([groups[p] for p in draw],ignore_index=True)
        for mid in ids:
            spec=freeze['catalog'][mid]
            obj=load_model(mid) if rep<0 else fit(sample,spec,f'refit_{rep:02d}__'+mid.replace('|','__'))
            c,j=effect_tables(obj,reference,plan[mid]['features'],plan[mid]['pairs'])
            if not c.empty:c['model_id']=mid;c['refit']=rep;rows.append(c)
            if not j.empty:j['model_id']=mid;j['refit']=rep;inter.append(j)
    table(pd.concat(rows,ignore_index=True),'mechanism_probe_curves');table(pd.concat(inter,ignore_index=True),'mechanism_probe_interactions')
    checkpoint('probes',[OUT/'tables/mechanism_probe_curves.csv',OUT/'tables/mechanism_probe_interactions.csv',OUT/'mechanism_probe_plan.json'])

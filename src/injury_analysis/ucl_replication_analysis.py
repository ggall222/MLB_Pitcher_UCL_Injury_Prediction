"""Original contrasts, whole-pitcher uncertainty and event composition sensitivity."""
import numpy as np
import pandas as pd
from src.evaluation import rank_metrics
from .ucl_dynamic_evaluation import metrics
from .ucl_temporal_data import interval_target
from .ucl_temporal_analysis import bootstrap_ap,uncertainty
from .ucl_replication_common import CFG,OUT,CACHE,OLD,table,read,parquet,checkpoint,valid
from .ucl_replication_data import prepare


def event_contributions(g,score):
    """Exact tie-aware AP decomposition, not a causal attribution or LOO estimate."""
    s=np.asarray(score);order=np.argsort(-s,kind='stable');y=g.target.to_numpy(dtype=int)[order];ss=s[order]
    if not y.sum():return pd.DataFrame(columns=['pitcher_id','dynamic_next_event_id','positive_rows','ap_contribution','fraction_total_ap'])
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    precision=np.cumsum(y)[ends]/(ends+1);perrow=np.repeat(precision,np.diff(np.r_[-1,ends]))
    q=g.iloc[order].copy();q['ap_contribution']=y*perrow/y.sum()
    cc=q.loc[q.target.eq(1)].groupby(['pitcher_id','dynamic_next_event_id']).agg(positive_rows=('target','size'),ap_contribution=('ap_contribution','sum')).reset_index()
    cc['fraction_total_ap']=cc.ap_contribution/cc.ap_contribution.sum();return cc


def leave_injured_pitcher_out(g,score):
    """Remove all evaluated rows for each injured pitcher, without refitting."""
    s=np.asarray(score);rows=[];original=rank_metrics(g.target,s)['average_precision']
    for pid in g.loc[g.target.eq(1),'pitcher_id'].unique():
        keep=g.pitcher_id.ne(pid);q=g.loc[keep];r=rank_metrics(q.target,s[keep]);prev=float(q.target.mean())
        rows.append(dict(pitcher_id=pid,rows=len(q),positive_rows=int(q.target.sum()),
            remaining_events=q.loc[q.target.eq(1),'dynamic_next_event_id'].nunique(),prevalence=prev,
            ap_without_pitcher=r['average_precision'],lift_without_pitcher=r['average_precision']/prev if prev else np.nan,
            original_ap=original,ap_drop=original-r['average_precision']))
    return pd.DataFrame(rows)


def contrast_table(matrix,boot,groups):
    """Unchanged near-minus-far normalized ablation contrasts and joint max-t bands."""
    rows=[];draws=[];bf,pf=boot['31-60']
    for family in groups:
        without='Without | '+family
        for near in ['1-7','8-14','15-30']:
            bn,pn=boot[near];d=(bn['Full union']-bn[without])/pn-(bf['Full union']-bf[without])/pf
            point=float(matrix.loc[matrix.family.eq(family)&matrix.interval.eq(near),'ablation_lift_loss'].iloc[0]-matrix.loc[matrix.family.eq(family)&matrix.interval.eq('31-60'),'ablation_lift_loss'].iloc[0])
            rows.append(dict(family=family,contrast=near+' minus 31-60',ablation_lift_difference=point,**uncertainty(d)));draws.append(d)
    draws=np.array(draws);points=np.array([r['ablation_lift_difference'] for r in rows]);se=np.nanstd(draws,axis=1,ddof=1)
    z=np.divide(draws-points[:,None],se[:,None],out=np.zeros_like(draws),where=se[:,None]>0)
    crit=float(np.nanquantile(np.nanmax(np.abs(z),axis=0),.95))
    for i,row in enumerate(rows):row.update(simultaneous_low=points[i]-crit*se[i],simultaneous_high=points[i]+crit*se[i],comparisons=len(rows),simultaneous_critical_value=crit)
    return pd.DataFrame(rows)


def run_analysis():
    if valid('analysis'):return
    f,e,spec=prepare();base=f.loc[f.target.notna()].copy()
    sets=[set(interval_target(base,lo,hi).loc[lambda z:z.target.eq(1),'dynamic_next_event_id']) for lo,hi in CFG['intervals']]
    shared=set.intersection(*sets);first=base.dynamic_next_target&base.dynamic_lead_days.between(1,60)
    shared_mask=(~first|base.dynamic_next_event_id.isin(shared)).to_numpy()
    rows=[];boot={};sharedboot={};matrix=[];sharedmatrix=[];sharedrows=[];contributions=[];influence=[];stable=[];bootrows=[]
    for lo,hi in CFG['intervals']:
        interval=f'{lo}-{hi}';g=interval_target(base,lo,hi).reset_index(drop=True)
        p=pd.read_parquet(CACHE/f'predictions_{interval}.parquet').set_index('row_id').loc[g.row_id]
        predictions={n:p[n].to_numpy() for n in spec['models']}
        b,prev=bootstrap_ap(g,predictions);boot[interval]=(b,prev)
        sb,sp=bootstrap_ap(g.loc[shared_mask],{n:s[shared_mask] for n,s in predictions.items()});sharedboot[interval]=(sb,sp)
        bootrows.append(pd.DataFrame(b).assign(prevalence=prev,interval=interval,replicate=np.arange(len(prev))))
        mr={};sm={}
        for name,s in predictions.items():
            m=metrics(g,s);mr[name]=m
            row=dict(interval=interval,model=name,**m)
            for prefix,d in [('ap',b[name]),('lift',b[name]/prev)]:row.update({prefix+'_'+k:v for k,v in uncertainty(d).items()})
            rows.append(row)
            sub=g.loc[shared_mask];q=metrics(sub,s[shared_mask]);sm[name]=q
            sharedrows.append(dict(interval=interval,model=name,shared_events=len(shared),**q,
                **{'lift_'+k:v for k,v in uncertainty(sb[name]/sp).items()}))
            cc=event_contributions(g,s);cc=cc.assign(interval=interval,model=name);contributions.append(cc)
            li=leave_injured_pitcher_out(g,s).assign(interval=interval,model=name);influence.append(li)
            top=cc.sort_values('ap_contribution',ascending=False).iloc[0] if len(cc) else None
            stable.append(dict(interval=interval,model=name,represented_events=m['represented_events'],injured_pitchers=len(li),
                original_ap=m['average_precision'],original_lift=m['pr_lift'],
                largest_event_id=top.dynamic_next_event_id if top is not None else None,
                largest_event_ap_contribution=top.ap_contribution if top is not None else np.nan,
                largest_event_fraction_ap=top.fraction_total_ap if top is not None else np.nan,
                loo_ap_min=li.ap_without_pitcher.min(),loo_ap_max=li.ap_without_pitcher.max(),
                loo_lift_min=li.lift_without_pitcher.min(),loo_lift_max=li.lift_without_pitcher.max(),
                maximum_pitcher_ap_drop=li.ap_drop.max()))
        for family in spec['families']:
            only='Only | '+family;without='Without | '+family
            for dest,mm,bb,pp in [(matrix,mr,b,prev),(sharedmatrix,sm,sb,sp)]:
                loss=mm['Full union']['average_precision']-mm[without]['average_precision'];pre=mm[only]['prevalence']
                dest.append(dict(interval=interval,family=family,standalone_lift=mm[only]['pr_lift'],
                    ablation_ap_loss=loss,ablation_lift_loss=loss/pre,**{'ablation_lift_'+k:v for k,v in uncertainty((bb['Full union']-bb[without])/pp).items()}))
        print('Locked replication bootstrap and event influence:',interval,flush=True)
    matrix=pd.DataFrame(matrix);smatrix=pd.DataFrame(sharedmatrix)
    contrasts=contrast_table(matrix,boot,spec['families']);sharedcontrasts=contrast_table(smatrix,sharedboot,spec['families'])
    old=pd.read_csv(OLD/'tables/temporal_contrasts_validation.csv')
    transfer=pd.read_csv(OLD/'tables/temporal_contrasts_test.csv')
    compare=contrasts.merge(old[['family','contrast','ablation_lift_difference']].rename(columns={'ablation_lift_difference':'development_effect'}),on=['family','contrast'])
    compare=compare.merge(transfer[['family','contrast','ablation_lift_difference']].rename(columns={'ablation_lift_difference':'transfer_effect'}),on=['family','contrast'])
    compare['same_direction_development']=np.sign(compare.ablation_lift_difference)==np.sign(compare.development_effect)
    compare['same_direction_transfer']=np.sign(compare.ablation_lift_difference)==np.sign(compare.transfer_effect)
    compare['absolute_effect_difference']=abs(compare.ablation_lift_difference-compare.development_effect)
    compare['effect_ratio_to_development']=compare.ablation_lift_difference/compare.development_effect.replace(0,np.nan)
    compare['simultaneously_supported']=compare.simultaneous_low.gt(0)|compare.simultaneous_high.lt(0)
    refs=[];met=pd.DataFrame(rows)
    for name in ['Departure','Profile','Background','Previous 62','Stable 62']:
        bn,pn=boot['1-7'];bf,pf=boot['31-60'];point=met.loc[met.model.eq(name)].set_index('interval').pr_lift
        refs.append(dict(model=name,contrast='1-7 minus 31-60 PR lift',lift_difference=point['1-7']-point['31-60'],**uncertainty(bn[name]/pn-bf[name]/pf)))
    table(met,'replication_metrics');table(matrix,'family_ablation');table(compare,'temporal_contrasts')
    table(pd.DataFrame(refs),'reference_temporal_contrasts');table(pd.DataFrame(sharedrows),'shared_event_sensitivity')
    table(smatrix,'shared_family_ablation');table(sharedcontrasts,'shared_temporal_contrasts')
    table(pd.concat(contributions,ignore_index=True),'event_ap_contributions');table(pd.concat(influence,ignore_index=True),'leave_one_injured_pitcher_out')
    table(pd.DataFrame(stable),'event_influence_summary');parquet(pd.concat(bootrows,ignore_index=True),'bootstrap_replication')
    checkpoint('analysis',[OUT/'tables'/(n+'.csv') for n in ['replication_metrics','family_ablation','temporal_contrasts',
        'reference_temporal_contrasts','shared_event_sensitivity','shared_family_ablation','shared_temporal_contrasts',
        'event_ap_contributions','leave_one_injured_pitcher_out','event_influence_summary']]+[CACHE/'bootstrap_replication.parquet'])

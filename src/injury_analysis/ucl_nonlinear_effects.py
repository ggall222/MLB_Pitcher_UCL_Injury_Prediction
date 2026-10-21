"""Training-supported local log-odds effects and interaction diagnostics.

The 1D curves are binned accumulated local effects. The 2D diagnostic is a
local mixed contrast, NOT a full second-order ALE or causal interaction estimate.
"""
import numpy as np
import pandas as pd
from .ucl_nonlinear_common import CFG
from .ucl_nonlinear_models import scores

def reference_sample(train):
    """Bound repeated-outing dominance without consulting targets."""
    rng=np.random.default_rng(CFG['seed']);f=train.copy();f['_order']=rng.random(len(f))
    f=f.sort_values('_order').groupby('pitcher_id',sort=False).head(CFG['reference_max_outings_per_pitcher'])
    return f.sort_values('_order').head(CFG['reference_rows']).drop(columns='_order').copy()

def training_edges(reference,feature,bins):
    x=reference[feature].dropna().to_numpy(dtype=float)
    return np.unique(np.quantile(x,np.linspace(.01,.99,bins+1))) if len(x) else np.array([])

def ale_1d(reference,feature,score_fn,edges):
    """Centered local differences on supported observed (non-imputed) values."""
    if len(edges)<3:return pd.DataFrame()
    f=reference.loc[reference[feature].between(edges[0],edges[-1])].copy()
    f['_bin']=np.clip(np.searchsorted(edges,f[feature],side='right')-1,0,len(edges)-2)
    rows=[];increments=[]
    for j in range(len(edges)-1):
        g=f.loc[f._bin.eq(j)]
        if g.empty:return pd.DataFrame()  # no invented connection through an empty bin
        low=g.copy();high=g.copy();low[feature]=edges[j];high[feature]=edges[j+1]
        delta=float(np.mean(score_fn(high)-score_fn(low)));increments.append(delta)
        rows.append(dict(feature=feature,bin=j,x=(edges[j]+edges[j+1])/2,lower=edges[j],upper=edges[j+1],
            rows=len(g),pitchers=g.pitcher_id.nunique(),local_difference=delta))
    out=pd.DataFrame(rows);values=np.cumsum(increments)-np.array(increments)/2
    out['ale_log_odds']=values-np.average(values,weights=out.rows)
    return out

def local_interactions(reference,left,right,score_fn,xedges,yedges):
    """Four local corners remove arbitrary additive main effects on log-odds.

    Joint occupancy is reported/masked. Even supported rectangles are model
    perturbations and may miss narrower correlations within each rectangle.
    """
    if len(xedges)<3 or len(yedges)<3:return pd.DataFrame()
    f=reference.loc[reference[left].between(xedges[0],xedges[-1])&reference[right].between(yedges[0],yedges[-1])].copy()
    f['_x']=np.clip(np.searchsorted(xedges,f[left],side='right')-1,0,len(xedges)-2)
    f['_y']=np.clip(np.searchsorted(yedges,f[right],side='right')-1,0,len(yedges)-2)
    records=[]
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            g=f.loc[f._x.eq(i)&f._y.eq(j)];supported=len(g)>=CFG['interaction_min_rows'] and g.pitcher_id.nunique()>=CFG['interaction_min_pitchers']
            contrast=np.nan
            if supported:
                parts=[]
                for x,y,sign in [(i+1,j+1,1),(i+1,j,-1),(i,j+1,-1),(i,j,1)]:
                    perturbed=g.copy();perturbed[left]=xedges[x];perturbed[right]=yedges[y];parts.append(sign*score_fn(perturbed))
                contrast=float(np.mean(np.sum(parts,axis=0)))
            records.append(dict(left=left,right=right,x_bin=i,y_bin=j,x_low=xedges[i],x_high=xedges[i+1],y_low=yedges[j],y_high=yedges[j+1],
                rows=len(g),pitchers=g.pitcher_id.nunique(),supported=supported,mixed_log_odds=contrast))
    return pd.DataFrame(records)

def effect_tables(obj,reference,features,pairs):
    curves=[];interactions=[]
    score_fn=lambda f:scores(obj,f,logit=True)
    for c in features:
        edges=training_edges(reference,c,CFG['ale_bins']);curves.append(ale_1d(reference,c,score_fn,edges))
    for left,right in pairs:
        interactions.append(local_interactions(reference,left,right,score_fn,
            training_edges(reference,left,CFG['interaction_bins']),training_edges(reference,right,CFG['interaction_bins'])))
    return (pd.concat(curves,ignore_index=True) if curves else pd.DataFrame(),
        pd.concat(interactions,ignore_index=True) if interactions else pd.DataFrame())

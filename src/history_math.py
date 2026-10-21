"""Vectorized prior-day rolling windows; shared by historical builders."""
import numpy as np


def prior_ends(dates):
    values=np.asarray(dates,dtype='datetime64[ns]')
    return np.searchsorted(values,values,side='left')


def moments(values, ends, window):
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    valid=np.isfinite(x)
    # Center before cumulative sums to stabilize variance for near-constant spin/speed.
    center=np.array([x[np.flatnonzero(valid[:,j])[0],j] if valid[:,j].any() else 0. for j in range(x.shape[1])])
    x=x-center
    c=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(valid,axis=0)])
    s=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(np.where(valid,x,0),axis=0)])
    q=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(np.where(valid,x*x,0),axis=0)])
    left=np.maximum(0,ends-window)
    count=c[ends]-c[left];total=s[ends]-s[left];square=q[ends]-q[left]
    mean=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
    variance=np.divide(square-total*mean,count-1,out=np.full_like(total,np.nan),where=count>1)
    sd=np.sqrt(np.maximum(variance,0))
    return count,mean+center,sd


def slopes(values,ends,window=5):
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    if not len(x):return np.full((len(ends),x.shape[1]),np.nan)
    idx=np.maximum(ends[:,None]-window+np.arange(window)[None,:],0)
    y=x[idx];valid=np.isfinite(y)
    xx=np.broadcast_to(np.arange(window)[None,:,None],y.shape)
    n=valid.sum(axis=1);sx=np.where(valid,xx,0).sum(axis=1)
    sy=np.where(valid,y,0).sum(axis=1);sxy=np.where(valid,xx*y,0).sum(axis=1)
    sxx=np.where(valid,xx*xx,0).sum(axis=1)
    numerator=n*sxy-sx*sy;denominator=n*sxx-sx*sx
    result=np.divide(numerator,denominator,out=np.full_like(sy,np.nan),where=(denominator>0)&(n>=3))
    result[ends<window]=np.nan
    return result

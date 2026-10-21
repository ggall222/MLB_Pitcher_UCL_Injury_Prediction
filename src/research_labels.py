"""Episode-aware historical outcomes; future evidence stays exclusively in labels/audits."""
import re
import numpy as np
import pandas as pd
from .utils import stable_id
from .build_labels import LABEL_COLUMNS


def build_events(injuries, transactions, box, cfg):
    """Link dated UCL procedures to active public arm-IL episodes, without inventing onset."""
    observed={pid:g.game_date.sort_values().to_numpy(dtype='datetime64[ns]') for pid,g in box.groupby('pitcher_id')}
    episodes=[]
    for pid,g in transactions.dropna(subset=['player_id','transaction_date']).groupby('player_id'):
        active=None
        for t in g.sort_values(['transaction_date','transaction_id']).itertuples():
            desc=str(t.description or '')
            arm=bool(re.search(r'\belbow\b|\bucl\b|tommy john|ulnar collateral|\bforearm\b',desc,re.I))
            if t.is_il_placement and arm and active is None:
                active={'episode_id':stable_id('arm_il',int(pid),str(t.transaction_date.date())),
                        'pitcher_id':int(pid),'reference_date':t.transaction_date,'il_date':t.effective_date,
                        'onset_transaction_id':int(t.transaction_id),
                        'evidence_source_url':getattr(t,'source_url',f'https://statsapi.mlb.com/api/v1/transactions?transactionIds={int(t.transaction_id)}'),
                        'activation_date':pd.NaT,
                        'reentry_date':pd.NaT,'description':desc,
                        'onset_confidence':'elbow_public_report' if t.is_elbow_related else 'forearm_public_report'}
                episodes.append(active)
            if t.is_il_activation and active is not None and t.transaction_date>=active['reference_date']:
                active['activation_date']=t.transaction_date
                dates=observed.get(pid,np.array([],dtype='datetime64[ns]'))
                dates=dates[dates>=np.datetime64(t.transaction_date)]
                if len(dates):active['reentry_date']=pd.Timestamp(dates[0])
                active=None
    episode_columns=['episode_id','pitcher_id','reference_date','il_date','onset_transaction_id',
                     'activation_date','reentry_date','description','onset_confidence','evidence_source_url']
    episode_frame=pd.DataFrame(episodes,columns=episode_columns)
    for c in ['reference_date','il_date','activation_date','reentry_date']:
        episode_frame[c]=pd.to_datetime(episode_frame[c])
    by_pid={pid:g for pid,g in episode_frame.groupby('pitcher_id')}
    records=[]
    for row in injuries.itertuples():
        candidate=None
        eligible=bool(row.confirmed_ucl_surgery and pd.notna(row.pitcher_id) and pd.notna(row.surgery_date_lower))
        g=by_pid.get(row.pitcher_id,episode_frame.iloc[:0])
        if eligible and not g.empty:
            candidates=g[(g.reference_date<=row.surgery_date_upper)&
                         (g.activation_date.isna() | (g.activation_date>=row.surgery_date_lower))]
            # The same active arm episode may precede surgery by >180 days.
            if len(candidates):candidate=candidates.sort_values('reference_date').iloc[-1]
        if candidate is not None:
            event_id=stable_id('ucl_event',candidate.episode_id)
            ref=candidate.reference_date;basis=candidate.onset_confidence
            episode_id=candidate.episode_id;il=candidate.il_date;reentry=candidate.reentry_date
        else:
            event_id=stable_id('unresolved_ucl_event',row.injury_id)
            ref=pd.NaT;basis='surgery_only_unknown_onset';episode_id=None;il=pd.NaT;reentry=pd.NaT
        records.append({'injury_event_id':event_id,'source_injury_id':row.injury_id,'pitcher_id':row.pitcher_id,
                        'pitcher_name':row.player_name,'episode_id':episode_id,
                        'label_reference_date':ref,'onset_confidence':basis,'il_date':il,
                        'onset_transaction_id':candidate.onset_transaction_id if candidate is not None else None,
                        'evidence_source_url':candidate.evidence_source_url if candidate is not None else None,
                        'first_reported_symptom_date':pd.NaT,'diagnosis_date':row.diagnosis_date,
                        'last_healthy_appearance':pd.NaT,'surgery_date':row.surgery_date,
                        'surgery_date_lower':row.surgery_date_lower,'surgery_date_upper':row.surgery_date_upper,
                        'surgery_date_precision':row.surgery_date_precision,'reentry_date':reentry,
                        'procedure_category':row.procedure_category,'repeat_procedure':row.repeat_procedure,
                        'confirmed_ucl_surgery':row.confirmed_ucl_surgery,'position':row.position,
                        'identity_method':row.identity_method})
    procedures=pd.DataFrame(records)
    # One injury episode can have multiple procedure records; keep a separate procedure map.
    events=[]
    for event_id,g in procedures.groupby('injury_event_id',sort=False):
        out=g.iloc[0].to_dict();out['source_injury_ids']=';'.join(g.source_injury_id)
        out['procedure_count']=len(g)
        out['procedure_categories']=';'.join(sorted(set(g.procedure_category)))
        out['surgery_date_lower']=g.surgery_date_lower.min();out['surgery_date_upper']=g.surgery_date_upper.max()
        events.append(out)
    event_frame=pd.DataFrame(events)
    for c in ['label_reference_date','il_date','first_reported_symptom_date','diagnosis_date',
              'last_healthy_appearance','surgery_date','surgery_date_lower','surgery_date_upper','reentry_date']:
        event_frame[c]=pd.to_datetime(event_frame[c])
    return episode_frame,event_frame,procedures


def build(appearances, events, episodes, box, cfg, sources_complete=True):
    end=pd.Timestamp(cfg['observation_end']);start=pd.Timestamp(cfg['start_date'])
    event_groups={pid:g for pid,g in events.dropna(subset=['pitcher_id']).groupby('pitcher_id')}
    episode_groups={pid:g.sort_values('reference_date') for pid,g in episodes.groupby('pitcher_id')}
    boxes={pid:g.game_date.sort_values().to_numpy(dtype='datetime64[ns]') for pid,g in box.groupby('pitcher_id')}
    output=[]
    for pid,a in appearances.groupby('pitcher_id',sort=False):
        ev=event_groups.get(pid,events.iloc[:0])
        ev=ev[ev.confirmed_ucl_surgery & ev.surgery_date_upper.ge(start)]
        known=ev[ev.label_reference_date.notna() & ev.label_reference_date.le(end)].sort_values('label_reference_date')
        uncertain=ev[ev.label_reference_date.isna()]
        uncertainty=[(r.surgery_date_lower-pd.Timedelta(days=cfg['uncertain_onset_lookback_days']),r.surgery_date_upper)
                     for r in uncertain.itertuples() if pd.notna(r.surgery_date_lower)]
        ep=episode_groups.get(pid,episodes.iloc[:0]);eplist=list(ep.itertuples());eventlist=list(known.itertuples())
        dates=boxes.get(pid,np.array([],dtype='datetime64[ns]'));dates=dates[dates<=np.datetime64(end)]
        last=pd.Timestamp(dates[-1]) if len(dates) else pd.NaT
        for row in a.itertuples():
            day=row.game_date
            active=[e for e in eplist if e.reference_date<=day and (pd.isna(e.reentry_date) or day<e.reentry_date)]
            later=[e for e in eplist if e.reference_date>day]
            stop=active[0].reference_date if active else (later[0].reference_date if later else pd.NaT)
            future=[e for e in eventlist if e.label_reference_date>day]
            event=future[0] if future else None
            event_date=event.label_reference_date if event else pd.NaT
            usable=event is not None and (pd.isna(stop) or event_date==stop)
            unknown_current=any(left<=day<=right for left,right in uncertainty)
            eligible=not active and not unknown_current and sources_complete
            out={'game_id':row.game_id,'pitcher_id':pid,'label_eligible':bool(eligible),
                 'post_known_injury':bool(active),'uncertain_injury_timing':unknown_current,
                 'risk_set_stop_date':stop,'next_known_ucl_onset':event_date,
                 'next_injury_id':event.injury_event_id if event else None,
                 'next_injury_event_id':event.injury_event_id if event else None,
                 'last_observed_appearance':last,'label_observation_end':end,
                 'label_sources_complete':bool(sources_complete),
                 'label_onset_basis':event.onset_confidence if event else None}
            for horizon in (14,30,60,90):
                limit=day+pd.Timedelta(days=horizon);value=None
                uncertain_horizon=any(left<=limit and right>day for left,right in uncertainty)
                if eligible:
                    if usable and event_date<=limit:
                        value=True
                    elif limit<=end and pd.notna(last) and limit<=last and (pd.isna(stop) or limit<stop) and not uncertain_horizon:
                        value=False
                out[f'ucl_injury_next_{horizon}d']=value
            cap=min(day+pd.Timedelta(days=cfg['followup_days']),end)
            future_dates=dates[(dates>np.datetime64(day))&(dates<=np.datetime64(cap))]
            for n in (3,5):
                value=None
                if eligible:
                    before_event=int(((dates>np.datetime64(day))&(dates<np.datetime64(event_date))).sum()) if event else None
                    if usable and event_date<=cap and before_event<n:value=True
                    elif len(future_dates)>=n:
                        nth=pd.Timestamp(future_dates[n-1])
                        if (pd.isna(stop) or nth<stop) and not any(left<=nth and right>day for left,right in uncertainty):value=False
                out[f'ucl_injury_next_{n}_appearances']=value
            output.append(out)
    df=pd.DataFrame(output)
    for c in LABEL_COLUMNS+['label_eligible','post_known_injury','uncertain_injury_timing','label_sources_complete']:
        df[c]=df[c].astype('boolean')
    for c in ['risk_set_stop_date','next_known_ucl_onset','last_observed_appearance','label_observation_end']:
        df[c]=pd.to_datetime(df[c])
    return df

"""Evidence-based UCL episode adjudication, separate from predictor knowledge time.

No procedure date is substituted for injury onset. 'Confirmed' below means
documented in these sources, not an independent clinical adjudication.
"""
import json
import numpy as np
import pandas as pd
from .ucl_dynamic_common import ROOT,CACHE,OUT,CFG,table,parquet,checkpoint,valid_checkpoint,partition_dates

ARM_COMPATIBLE={'Elbow / UCL','Forearm','Upper Arm','Unknown / Unspecified'}

def parse_list(text):
    if isinstance(text,list):return text
    if not isinstance(text,str):return []
    try:return json.loads(text)
    except (ValueError,TypeError):return []

def validate_procedure_link(event,procedure,appearances):
    """Validate an existing identifier-based link; never match names or remote dates."""
    reasons=[];ref=event.anchor_date
    lo=pd.to_datetime(procedure.surgery_date_lower);hi=pd.to_datetime(procedure.surgery_date_upper)
    if pd.isna(ref) or pd.isna(lo) or pd.isna(hi):return False,'missing interval or event date'
    if int(procedure.player_id)!=int(event.player_id):reasons.append('identifier mismatch')
    if procedure.surgery_source_level!='MLB':reasons.append('procedure source not MLB level')
    if lo<ref or hi>ref+pd.Timedelta(days=CFG['procedure_link_max_days']):reasons.append('procedure outside bounded post-placement interval')
    if event.activation_match_valid and pd.notna(event.activation_date) and hi>=event.activation_date:reasons.append('activation precedes procedure')
    if pd.notna(event.episode_boundary_date) and hi>event.episode_boundary_date:reasons.append('procedure beyond episode boundary')
    # A clearly observed return after the report breaks this conservative link.
    if len(appearances):
        dates=appearances.game_date
        if ((dates>event.public_known_date)&(dates<lo)).any():reasons.append('MLB appearance between public placement and procedure')
    if event.initial_injury_region not in ARM_COMPATIBLE:reasons.append('initial anatomy incompatible')
    return not reasons,'; '.join(reasons)

def classify_evidence(event,procedure_supported,probable_supplement,official_current_ucl=None):
    official_current_ucl=event.explicit_ucl if official_current_ucl is None else official_current_ucl
    if official_current_ucl and event.injury_region=='Elbow / UCL':
        return 'confirmed_official_documentation'
    if procedure_supported:return 'confirmed_procedure_supported'
    if probable_supplement:return 'probable_supplemental_documentation'
    if event.initial_injury_region=='Elbow / UCL':
        return 'ambiguous_elbow' if event.diagnosis_uncertain or event.initial_injury_diagnosis=='Unknown / Unspecified' else 'broader_elbow'
    if event.initial_injury_region=='Forearm':return 'forearm_without_ucl_confirmation'
    return 'other_or_unknown_injury'

def run_endpoint():
    if valid_checkpoint('endpoint'):return pd.read_parquet(CACHE/'adjudicated_events.parquet')
    e=pd.read_parquet(ROOT/'data/injury/analysis/events.parquet')
    appearances=pd.read_parquet(ROOT/'data/injury/analysis/appearances.parquet',columns=['pitcher_id','game_date','game_id'])
    by_pitcher={p:g for p,g in appearances.groupby('pitcher_id')}
    comparison=pd.read_csv(ROOT/'reports/general_injury/ucl_pipeline_comparison.csv')
    links=comparison.loc[comparison.comparison_status.eq('both')].copy()
    proc_by_event={eid:g for eid,g in links.groupby('general_injury_event_id')}
    associations=pd.read_csv(ROOT/'reports/general_injury/supplemental_source_matches.csv')
    association_index=associations.set_index('supplemental_record_id')
    tx=pd.read_parquet(ROOT/'data/injury/processed/mlb_pitcher_injury_transactions_2015_2026.parquet')
    txlinks=pd.read_csv(ROOT/'data/injury/processed/injury_event_transaction_links.csv')
    linked=txlinks.merge(tx,on=['transaction_id','player_id'],how='left',validate='many_to_one')
    official=linked.loc[linked.explicit_ucl.fillna(False)&linked.injury_region.eq('Elbow / UCL')]
    # A later recovery/procedure-only entry can refer to an older operation.
    # It is not independent evidence of a new UCL injury at this IL placement.
    current_official=official.loc[~official.surgery_flag.fillna(False)]
    first_ucl=current_official.groupby('injury_event_id').transaction_date.min()
    evidence_records=[];rows=[]
    for r in e.itertuples():
        procedures=proc_by_event.get(r.injury_event_id,pd.DataFrame())
        accepted=[]
        for p in procedures.itertuples():
            valid,reason=validate_procedure_link(r,p,by_pitcher.get(r.player_id,pd.DataFrame()))
            evidence_records.append(dict(injury_event_id=r.injury_event_id,player_id=r.player_id,
                evidence_id=p.existing_ucl_event_id,evidence_type='linked_ucl_procedure',valid=valid,reason=reason,
                source=p.surgery_source,evidence_date=p.surgery_date,match_method=p.match_method,
                date_lower=p.surgery_date_lower,date_upper=p.surgery_date_upper,
                historically_observable=False))
            if valid:accepted.append(p)
        supplemental=parse_list(r.supplemental_diagnoses)
        official_current=r.injury_event_id in first_ucl.index
        probable=(any(d.get('structure')=='UCL' and d.get('region')=='Elbow / UCL' for d in supplemental)
                  or (r.explicit_ucl and not official_current))
        category=classify_evidence(r,bool(accepted),probable,official_current)
        for t in official.loc[official.injury_event_id.eq(r.injury_event_id)].itertuples():
            evidence_records.append(dict(injury_event_id=r.injury_event_id,player_id=r.player_id,
                evidence_id=str(t.transaction_id),evidence_type='official_ucl_text',valid=not t.surgery_flag,
                reason='Procedure/recovery-only text requires a validated procedure link' if t.surgery_flag else '',
                source=t.source_url,evidence_date=t.transaction_date,match_method='canonical episode transaction ID',
                date_lower=t.transaction_date,date_upper=t.transaction_date,
                historically_observable=False))
        # Elbow-versus-forearm progression is compatible with UCL adjudication;
        # unrelated anatomical disagreement remains excluded. This is explicit,
        # unlike silently clearing all supplemental-source conflicts.
        compatible_progression=(r.initial_injury_region in ARM_COMPATIBLE and (bool(accepted) or r.explicit_ucl) and
            all(d.get('region') in ARM_COMPATIBLE for d in supplemental if d.get('region')))
        reasons=[]
        for flag in ['missing_initial_date','left_censored','uncertain_timing','overlapping_episode',
                     'multiple_regions','postoperative_or_preexisting','inferred_continuation','identity_review','incomplete_sources']:
            if getattr(r,'exclude_'+flag):reasons.append(flag)
        if r.exclude_anatomy_source_conflict and not compatible_progression:reasons.append('unresolved_anatomy_conflict')
        if r.initial_injury_region not in ARM_COMPATIBLE:reasons.append('incompatible_initial_anatomy')
        if pd.notna(r.throwing_arm_injury) and not r.throwing_arm_injury:reasons.append('explicit_nonthrowing_side')
        if r.player_id not in by_pitcher:reasons.append('no_regular_season_pitcher_history')
        # Ambiguous procedure dates can support event identity but are never onset.
        outcome_known_at=first_ucl.get(r.injury_event_id,pd.NaT)
        if pd.isna(outcome_known_at) and (accepted or probable):
            outcome_known_at=r.supplemental_known_at if pd.notna(r.supplemental_known_at) else r.surgery_known_at
        d=r._asdict();d.update(evidence_category=category,procedure_link_valid=bool(accepted),
            linked_procedure_ids=';'.join(str(p.existing_ucl_event_id) for p in accepted),
            accepted_procedure_count=len(accepted),probable_supplemental_ucl=probable,
            official_current_ucl_documented=official_current,
            compatible_forearm_elbow_progression=bool(r.exclude_anatomy_source_conflict and compatible_progression),
            ucl_confirmation_reported_date=outcome_known_at,
            prediction_reference_date=r.anchor_date,reference_date_basis='earliest canonical onset/retroactive/effective and reported-placement proxy',
            first_public_indication_date=r.public_known_date,diagnosis_date=first_ucl.get(r.injury_event_id,pd.NaT),
            timing_basis=r.injury_event_date_basis,timing_uncertainty_days=(r.public_known_date-r.anchor_date).days if pd.notna(r.anchor_date) else np.nan,
            biological_onset_established=bool(pd.notna(r.explicitly_documented_onset_date)),
            source_publication_archive_verified=bool(r.historical_publication_verified),
            postoperative_entry=bool(r.exclude_postoperative_or_preexisting),
            endpoint_exclusions=';'.join(reasons),timing_usable=not reasons)
        d['endpoint_primary']=not reasons and category.startswith('confirmed_')
        d['endpoint_strict_official']=not reasons and category=='confirmed_official_documentation'
        d['endpoint_expanded']=not reasons and (category.startswith('confirmed_') or category.startswith('probable_'))
        # Unknown/UCL-adjacent episodes cannot be silently negative labels.
        d['uncertain_ucl_competitor']=(r.initial_injury_region in ARM_COMPATIBLE or r.explicit_ucl or bool(accepted) or probable)
        rows.append(d)
    result=pd.DataFrame(rows);result['endpoint_partition']=partition_dates(result.prediction_reference_date)
    # One procedure may support only one canonical primary episode.
    exploded=result.loc[result.endpoint_primary,['injury_event_id','linked_procedure_ids']].assign(
        procedure=lambda x:x.linked_procedure_ids.str.split(';')).explode('procedure')
    duplicates=set(exploded.loc[exploded.procedure.ne('')&exploded.procedure.duplicated(False),'injury_event_id'])
    if duplicates:
        mask=result.injury_event_id.isin(duplicates)
        result.loc[mask,['endpoint_primary','endpoint_strict_official','endpoint_expanded','timing_usable']]=False
        result.loc[mask,'endpoint_exclusions']+=';shared_procedure_episode_ambiguity'
    table(pd.DataFrame(evidence_records),'ucl_evidence_links')
    table(result,'ucl_endpoint_adjudication')
    table(result.loc[(result.explicit_ucl|result.supplemental_ucl_confirmed|result.probable_supplemental_ucl)&~result.endpoint_primary], 'ucl_excluded_or_uncertain')
    table(comparison.loc[comparison.comparison_status.eq('existing_ucl_only')],'unlinked_legacy_ucl_procedures')
    table(result.groupby(['endpoint_partition','evidence_category']).agg(episodes=('injury_event_id','size'),
        usable_primary=('endpoint_primary','sum'),usable_strict=('endpoint_strict_official','sum'),
        usable_expanded=('endpoint_expanded','sum')).reset_index(),'endpoint_counts')
    table(result.loc[result.endpoint_primary].groupby('season').agg(events=('injury_event_id','size'),
        pitchers=('player_id','nunique'),initial_explicit=('initial_explicit_ucl','sum'),
        procedure_supported=('procedure_link_valid','sum'),public_date_gap_median=('timing_uncertainty_days','median')).reset_index(),'ucl_events_by_season')
    parquet(result,'adjudicated_events')
    checkpoint('endpoint',[CACHE/'adjudicated_events.parquet',OUT/'tables/ucl_endpoint_adjudication.csv',OUT/'tables/ucl_evidence_links.csv'])
    print(f'UCL adjudication: {int(result.endpoint_primary.sum())} primary; {int(result.endpoint_strict_official.sum())} strict official; {int(result.endpoint_expanded.sum())} expanded.',flush=True)
    return result

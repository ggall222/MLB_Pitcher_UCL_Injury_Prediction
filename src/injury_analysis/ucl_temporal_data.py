"""Non-overlapping first-event outcomes on a common, observed 60-day risk set."""
import json
import pandas as pd
from .ucl_dynamic_data import prepare_data, risk_frame, event_cohort
from .ucl_temporal_common import ROOT, OUT, CACHE, CFG, snapshot, table, parquet, write_json, checkpoint, valid_checkpoint, interval_name

def interval_target(frame, lo, hi):
    """Other first-UCL intervals are negatives for THIS interval, not healthy claims.

    Unknown first-event outcomes remain unknown. Never look past an intervening
    injury. For the primary analysis, input is the same 60-day risk frame.
    """
    if not (1 <= lo <= hi <= int(frame.horizon.iloc[0])):
        raise ValueError('Interval exceeds available observation horizon')
    f = frame.copy()
    f['cumulative_target'] = f.target
    f['target'] = (f.dynamic_next_target & f.dynamic_lead_days.between(lo, hi)).astype('Int8')
    f.loc[f.cumulative_target.isna(), 'target'] = pd.NA
    f['interval'] = interval_name(lo, hi)
    return f

def definitions(specs):
    inv = pd.read_csv(ROOT/'reports/feature_consolidation/tables/feature_inventory.csv')
    features = sorted({c for name in CFG['representations'] for c in specs[name]['features']})
    selected = inv.loc[inv.feature.isin(features)].copy()
    if set(selected.feature) != set(features) or selected.feature.duplicated().any():
        raise ValueError('Every frozen field must have exactly one inventory definition')
    if not selected.eligible_predictor.all():
        raise ValueError('Frozen union includes a previously excluded field')
    groups = {k:g.feature.tolist() for k,g in selected.groupby('feature_group', sort=True)}
    models = {'Full union': features}
    for family, fields in groups.items():
        models['Only | '+family] = fields
        models['Without | '+family] = [c for c in features if c not in fields]
    models.update({name:specs[name]['features'] for name in CFG['representations']})
    return selected, groups, models

def support(frame, events, policy):
    records = []
    for lo, hi in CFG['intervals']:
        f = interval_target(frame, lo, hi)
        for part in ['train', 'validation', 'test']:
            allrows = f.loc[f.partition.eq(part) & f.analysis_eligible]
            g = allrows.loc[allrows.target.notna()]
            positive = g.loc[g.target.eq(1)]
            ev = event_cohort(events, part)
            represented = positive.dynamic_next_event_id.nunique()
            opportunities = positive.loc[positive.dynamic_next_event_id.isin(ev.injury_event_id)]
            records.append(dict(risk_policy=policy, partition=part, interval=interval_name(lo, hi),
                eligible_landmarks=len(allrows), rows=len(g), pitchers=g.pitcher_id.nunique(),
                unknown_rows=int(allrows.target.isna().sum()), positive_rows=int(g.target.sum()),
                prevalence=float(g.target.mean()), represented_events=represented,
                events_with_opportunity=opportunities.dynamic_next_event_id.nunique(),
                qualifying_events=len(ev), events_without_opportunity=len(ev)-opportunities.dynamic_next_event_id.nunique(),
                other_interval_ucl_rows=int((g.cumulative_target.eq(1)&g.target.eq(0)).sum()),
                known_competing_rows=int(g.label_status.eq('non_ucl_competing_first_event').sum())))
    return pd.DataFrame(records)

def prepare():
    if valid_checkpoint('data'):
        return (pd.read_parquet(CACHE/'risk_60.parquet'),
                pd.read_parquet(ROOT/'data/injury/analysis/ucl_dynamic/adjudicated_events.parquet'),
                json.loads((CACHE/'families.json').read_text()),
                json.loads((CACHE/'model_definitions.json').read_text()))
    snapshot()
    a, events, specs = prepare_data()
    inventory, groups, models = definitions(specs)
    f = risk_frame(a, events, horizon=60)
    table(inventory, 'feature_inventory')
    table(pd.DataFrame([dict(family=k, fields=len(v), features=';'.join(v)) for k,v in groups.items()]), 'family_definitions')
    prior = pd.read_csv(ROOT/'reports/feature_consolidation/tables/feature_inventory.csv')
    missing = prior.loc[~prior.feature_group.isin(groups)]
    table(missing[['feature', 'feature_group', 'screen_exclusion', 'classification_ambiguity',
                   'training_missing_fraction', 'support_only_sensitivity']], 'omitted_groups')
    table(support(f, events, 'common_60d'), 'cohort_support')
    table(f.loc[f.analysis_eligible].groupby(['partition','label_status']).size().rename('rows').reset_index(), 'censoring')
    parquet(f, 'risk_60')
    write_json(groups, CACHE/'families.json'); write_json(models, CACHE/'model_definitions.json')
    write_json(dict(config=CFG, families=groups, models=models, source='Unchanged union of five frozen representations',
        estimand='First UCL episode in a disjoint interval after a completed appearance, before another injury',
        primary_risk_set='Identical observed 60-day first-event cohort for every interval',
        negative_definition='Known non-UCL first event, first UCL in another interval, or observed event-free 60 days',
        uncertainty='Paired whole-pitcher validation bootstrap; separate training refits for development-nominated families',
        permutation='Joint family row permutation within season; marginal reliance, not causal/conditional importance',
        selection='At most two development candidates by mean standardized ablation loss, with signs and interval uncertainty retained',
        transfer='No transfer analysis before development freeze; no independent confirmation',
        sensitivity='Interval-upper-bound follow-up for Full union, Profile and Departure; no new tuning'),
        OUT/'predeclared_analysis.json')
    checkpoint('data', [CACHE/'risk_60.parquet', CACHE/'families.json', CACHE/'model_definitions.json',
                       OUT/'tables/feature_inventory.csv', OUT/'tables/cohort_support.csv', OUT/'predeclared_analysis.json'])
    return f, events, groups, models

"""Read-only diagnostic: exact expected event recall for uniformly random alerts.

Uses the same category comparison cohort, fixed alert count, 7–30-day warning
window and all-event denominator as the existing warning-budget figure.
No model fitting, no change to original metrics, and no database writes.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.injury_analysis.common import CACHE, CFG, OUT, source_signature


def probability_detected(n, k, opportunities):
    """1 - C(n-m,k)/C(n,k), evaluated using at most m factors."""
    if not 0 <= k <= n or not 0 <= opportunities <= n:
        raise ValueError('Need 0 <= alerts, opportunities <= eligible appearances')
    if opportunities == 0 or k == 0:
        return 0.0
    if opportunities > n - k:
        return 1.0
    return float(1 - np.prod([(n-k-j)/(n-j) for j in range(opportunities)]))


def main():
    signature = source_signature()
    predictions = pd.read_parquet(CACHE / 'test_predictions.parquet')
    events = pd.read_parquet(CACHE / 'events.parquet')
    actual = pd.read_csv(OUT / 'tables/alert_budget_metrics.csv').query('horizon == 30')
    rows = []
    for category, f in predictions.loc[predictions.horizon.eq(30)].groupby('category'):
        ev = events.loc[events.target_eligible & events.analysis_group.eq(category)
            & events.anchor_date.gt(CFG['validation_end']) & events.anchor_date.le(CFG['test_end'])]
        opportunities = f[['pitcher_id','game_id','game_date']].merge(
            ev[['player_id','injury_event_id','anchor_date']],left_on='pitcher_id',right_on='player_id')
        lead = (opportunities.anchor_date - opportunities.game_date).dt.days
        counts = opportunities.loc[lead.between(7,30)].groupby('injury_event_id').size().reindex(ev.injury_event_id,fill_value=0)
        for row in actual.loc[actual.category.eq(category)].itertuples():
            n, k = len(f), int(np.floor(row.alert_rate * len(f)))
            expected = float(np.mean([probability_detected(n,k,int(m)) for m in counts]))
            assert int(row.qualifying_events) == len(ev)
            assert int(row.alerts) == k
            rows.append(dict(category=category, horizon=30, minimum_warning_days=7, alert_rate=row.alert_rate,
                eligible_appearances=n, alerts=k, qualifying_events=len(ev),
                observable_events=int(counts.gt(0).sum()), observed_event_recall=row.injury_event_recall,
                expected_uniform_random_event_recall=expected,
                observed_minus_expected=row.injury_event_recall-expected,
                observed_detected_events=row.alerted_events, expected_detected_events=expected*len(ev)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'tables/random_alert_event_benchmark.csv',index=False)
    assert source_signature() == signature
    print(result.loc[result.alert_rate.eq(.05), ['category','observed_event_recall',
        'expected_uniform_random_event_recall','observed_minus_expected']].to_string(index=False))
    print('Exact expectations only; no confidence interval or superiority test is implied.')


if __name__ == '__main__':
    main()

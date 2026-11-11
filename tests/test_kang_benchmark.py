"""Paper adapter invariants: grain, season rules, label timing and nested selection."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.injury_analysis.kang_spec import EXTRA_SQL, FEATURE_NAMES, load_config
from src.injury_analysis.kang_cohort import season_eligibility, construct_cohorts, window_labels
from src.injury_analysis.kang_models import reserve_random_holdout, cluster_auc_ci


def appearances(pitchers=(1,), dates=('2019-05-01', '2019-05-05', '2019-05-10', '2019-05-15', '2019-05-20')):
    rows = []
    for pitcher in pitchers:
        for i, day in enumerate(dates):
            rows.append(dict({f: 1. for f in FEATURE_NAMES}, pitcher_id=pitcher, game_id=100 * pitcher + i,
                season=2019, game_date=pd.Timestamp(day), position='P', is_starter=True,
                outs_recorded=15, game_type='R'))
    return pd.DataFrame(rows)


def event(pitcher, date, region, **kw):
    return dict(dict(injury_event_id=str(pitcher) + date, pitcher_id=pitcher,
        transaction_date=pd.Timestamp(date), initial_injury_region=region,
        placement_transaction_id=pitcher, orphan_transfer=False, left_censored=False,
        overlapping_episode=False, timing_uncertain=False, multiple_injury_regions=False), **kw)


def test_season_thresholds_are_inclusive_and_not_five_prior_appearances():
    cfg = load_config()
    a = appearances(dates=pd.date_range('2019-05-01', periods=20))
    a.loc[:1, 'kang_spin_sd'] = np.nan  # exactly 10% of games have >=1 missing
    _, s = season_eligibility(a, cfg)
    assert s.eligible.all()
    a.loc[2, 'kang_spin_sd'] = np.nan
    assert not season_eligibility(a, cfg)[1].eligible.any()
    assert season_eligibility(appearances(), cfg)[1].eligible.all()
    assert not season_eligibility(appearances().iloc[:4], cfg)[1].eligible.any()


def test_missingness_counts_games_not_cells():
    a = appearances(dates=pd.date_range('2019-05-01', periods=20))
    a.loc[0, FEATURE_NAMES] = np.nan
    s = season_eligibility(a, load_config())[1]
    assert s.missing_fraction.iloc[0] == .05


def test_season_role_majority_innings_tie_and_unresolved():
    a = appearances(dates=pd.date_range('2019-05-01', periods=6))
    a['is_starter'] = [True] * 3 + [False] * 3
    a['outs_recorded'] = [9] * 3 + [3] * 3
    assert season_eligibility(a, load_config())[1].season_role.iloc[0] == 'starter'
    a['outs_recorded'] = 3
    assert not season_eligibility(a, load_config())[1].eligible.any()
    a['is_starter'] = [True] * 2 + [False] * 4
    assert season_eligibility(a, load_config())[1].season_role.iloc[0] == 'reliever'


def test_two_way_players_retained_and_emergency_position_players_excluded():
    a = appearances((1, 2))
    a.loc[a.pitcher_id.eq(1), 'position'] = 'TWP'
    a.loc[a.pitcher_id.eq(2), 'position'] = 'SS'
    assert set(season_eligibility(a, load_config())[0].pitcher_id) == {1}


def test_first_combined_endpoint_and_controls_do_not_use_later_diagnosis():
    a = appearances((1, 2, 3, 4, 5, 6))
    e = pd.DataFrame([
        event(1, '2019-05-12', 'Elbow / UCL'), event(1, '2019-05-18', 'Shoulder'),
        event(2, '2019-05-12', 'Shoulder'), event(4, '2019-05-12', 'Hamstring'),
        event(5, '2019-05-12', 'Unknown / Unspecified'),
        event(6, '2019-05-10', 'Elbow / UCL', overlapping_episode=True),
        event(6, '2019-05-18', 'Shoulder')])
    cohorts, _, first, _, _ = construct_cohorts(a, e, load_config())
    elbow = cohorts[('no_any_il', 'elbow', 'starter')]
    shoulder = cohorts[('no_any_il', 'shoulder', 'starter')]
    assert set(elbow.pitcher_id) == {1, 3}
    assert set(shoulder.pitcher_id) == {2, 3}
    assert set(cohorts[('no_shoulder_elbow_il', 'elbow', 'starter')].pitcher_id) == {1, 3, 4}
    assert not first[first.pitcher_id.eq(6)].eligible_case.any()  # no promotion of second event
    assert (elbow.loc[elbow.pitcher_id.eq(1), 'game_date'] < pd.Timestamp('2019-05-12')).all()
    y = window_labels(elbow, 7)
    assert y.sum() == 2
    assert y[elbow.game_date.eq('2019-05-01')].sum() == 0


def test_same_day_and_postinjury_games_are_excluded():
    a = appearances((1, 2))
    c, *_ = construct_cohorts(a, pd.DataFrame([event(1, '2019-05-10', 'Elbow / UCL')]), load_config())
    f = c[('no_any_il', 'elbow', 'starter')]
    assert f[f.pitcher_id.eq(1)].game_date.max() == pd.Timestamp('2019-05-05')
    assert len(f[f.pitcher_id.eq(2)]) == 5


def test_all_pitch_sd_and_release_y_are_not_fastball_sd_or_extension():
    p = pd.DataFrame(dict(release_speed=[100., 80., 20.], release_spin_rate=[2000., 2400., 9999.],
                          release_pos_x=[-2., 2., 9.], release_pos_y=[54., 55., 54.]))
    with duckdb.connect(':memory:') as c:
        c.register('raw_pitches', p)
        r = c.execute('SELECT ' + ','.join(v + ' AS ' + k for k, v in EXTRA_SQL.items()) + ' FROM raw_pitches').df().iloc[0]
    assert r.kang_velocity_sd == pytest.approx(np.std([100, 80], ddof=1))
    assert r.kang_spin_max == 2400
    assert r.kang_spin_sd == pytest.approx(np.std([2000, 2400], ddof=1))
    assert r.kang_release_abs_x_avg == 2  # abs(mean(X)) would wrongly return 0
    assert r.kang_release_y_avg > 50


def test_protect_2026_and_duplicate_appearance_grain(tmp_path):
    cfg = load_config()
    f = appearances()
    f['game_date'] = pd.Timestamp('2026-01-01')
    with pytest.raises(ValueError, match='2026'):
        season_eligibility(f, cfg)
    with pytest.raises(ValueError, match='2026'):
        reserve_random_holdout(f, cfg)
    with pytest.raises(ValueError, match='Duplicate'):
        season_eligibility(pd.concat([appearances(), appearances()]), cfg)
    import yaml
    cfg['end_date'] = '2026-01-01'
    p = tmp_path / 'bad.yaml'
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='2026'):
        load_config(p)


def test_cluster_bootstrap_reproducible_and_perfect_auc():
    f = pd.DataFrame(dict(pitcher_id=np.repeat(range(8), 3), season=2019))
    y = np.tile([1, 0, 0], 8)
    a = cluster_auc_ci(f, y, y * .8 + .1, 30, 12)
    assert a == (1., 1., 30)
    assert a == cluster_auc_ci(f, y, y * .8 + .1, 30, 12)


def test_episode_qa_cannot_be_revised_using_2026_transactions_or_outings(monkeypatch):
    from src.injury_analysis import kang_cohort as c
    tx = pd.DataFrame(dict(transaction_date=pd.to_datetime(['2025-12-01'])))
    observed = pd.DataFrame(dict(game_date=pd.to_datetime(['2025-09-01'])))
    def existing_builder(t, config, a):
        assert config['end_date'] == '2025-12-31'
        assert t.transaction_date.max().year == 2025
        return pd.DataFrame({'asof_safe': [True]}), None, None
    monkeypatch.setattr(c, 'build_episodes', existing_builder)
    assert c.bounded_episodes(tx, observed, load_config()).asof_safe.all()
    tx.loc[0, 'transaction_date'] = pd.Timestamp('2026-01-01')
    with pytest.raises(ValueError, match='cutoff'):
        c.bounded_episodes(tx, observed, load_config())
    tx.loc[0, 'transaction_date'] = pd.Timestamp('2025-12-01')
    observed.loc[0, 'game_date'] = pd.Timestamp('2026-04-01')
    with pytest.raises(ValueError, match='cutoff'):
        c.bounded_episodes(tx, observed, load_config())


def test_selection_is_fold_local_and_freeze_precedes_holdout(tmp_path, monkeypatch):
    from src.injury_analysis import kang_models as m
    f = appearances(pitchers=range(1, 31))
    f['avg_velocity'] = np.arange(len(f), dtype=float)
    f['is_case_season'] = np.arange(len(f)) % 3 == 0
    f['lead_days'] = np.where(f.is_case_season, 5, np.nan)
    f['injury_event_id'] = np.where(f.is_case_season, f.pitcher_id.astype(str), None)
    cfg = load_config()
    cfg.update(cache=str(tmp_path), windows_days=[7, 14], cv_folds=2, bootstrap_iterations=5)
    train, holdout = reserve_random_holdout(f, cfg)
    held_values = set(f.iloc[holdout].avg_velocity)
    training_calls = []

    class FakeModel:
        def predict_proba(self, x):
            if set(x.avg_velocity) & held_values:
                freeze = tmp_path / 'models/elbow_starter/freeze.json'
                assert freeze.exists()
                assert json.loads(freeze.read_text())['evaluated_2026'] is False
            p = (x.avg_velocity.to_numpy() + 1) / (len(f) + 2)
            return np.column_stack([1-p, p])

        def save_model(self, path):
            Path(path).write_text('test-only model')

    def select(x, y, config, seed):
        assert not (set(x.avg_velocity) & held_values)
        training_calls.append(set(x.avg_velocity))
        return ['avg_velocity'], pd.DataFrame([dict(feature='avg_velocity', retained=True)])

    def fit(x, y, config, seed):
        assert not (set(x.avg_velocity) & held_values)
        return FakeModel()

    monkeypatch.setattr(m, 'select_features', select)
    monkeypatch.setattr(m, 'fit_xgb', fit)
    result, cv, _ = m.run_task(f, cfg, 'elbow', 'starter', 'test')
    assert len(cv) == 4
    assert all(len(x) < len(train) for x in training_calls[:-1])
    assert len(training_calls[-1]) == len(train)
    assert result['holdout_rows'] == len(holdout)
    # Verified checkpoints skip every fit; corrupted model must not be reused.
    monkeypatch.setattr(m, 'fit_xgb', lambda *args: pytest.fail('valid checkpoint was not reused'))
    assert m.run_task(f, cfg, 'elbow', 'starter', 'test')[0] == result
    (tmp_path / 'models/elbow_starter/model.ubj').write_text('interrupted/corrupt output')
    monkeypatch.setattr(m, 'fit_xgb', fit)
    before = len(training_calls)
    m.run_task(f, cfg, 'elbow', 'starter', 'test')
    assert len(training_calls) == before * 2

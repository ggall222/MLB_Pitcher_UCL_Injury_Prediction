"""Train-only PCA, explicit profile/change blocks, and redundancy diagnostics.

Source features and preceding analysis outputs are read-only. PCA is unwhitened:
full-rank rotation preserves the existing isotropic L2 logistic penalty.
"""
import json
import warnings
import joblib
import duckdb
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import ROOT, CFG as ORIGINAL_CFG, CACHE as ORIGINAL_CACHE, source_signature, sha
from .signals_common import inputs as signal_inputs

CONFIG_PATH = ROOT / 'config/injury_reduction.yaml'
CFG = yaml.safe_load(CONFIG_PATH.read_text())
OUT = ROOT / CFG['output_path']
CACHE = ROOT / CFG['cache_path']
for folder in [OUT, OUT/'tables', OUT/'figures', CACHE]:
    folder.mkdir(parents=True, exist_ok=True)

# These refer to the SAME eight measurements. Means and standardized departures
# use existing strictly prior pitcher baselines, never a whole-career/future mean.
BASELINE_PAIRS = {
    'avg_fb_velo_baseline_mean': 'velo_z',
    'release_x_avg_baseline_mean': 'release_x_z',
    'release_z_avg_baseline_mean': 'release_z_z',
    'release_x_sd_baseline_mean': 'release_x_sd_z',
    'release_z_sd_baseline_mean': 'release_z_sd_z',
    'extension_avg_baseline_mean': 'extension_z',
    'spin_avg_baseline_mean': 'spin_z',
    'zone_pct_baseline_mean': 'zone_pct_z',
}
BINARY = ['throws_left', 'is_starter']


def table(frame, name):
    frame.to_csv(OUT/'tables'/f'{name}.csv', index=False)
    return frame


def read_table(name):
    return pd.read_csv(OUT/'tables'/f'{name}.csv')


def figure(fig, name):
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT/'figures'/f'{name}.{suffix}', bbox_inches='tight', dpi=220)
    return fig


def inputs():
    a, events, spec, matches = signal_inputs()
    missing = [c for c in BASELINE_PAIRS if c not in a]
    before = source_signature()
    if missing:
        with duckdb.connect(str(ROOT/ORIGINAL_CFG['database_path']), read_only=True) as con:
            available = set(con.execute('describe model_features').fetchdf().column_name)
            if not set(missing) <= available:
                raise ValueError('Existing pitcher baselines unavailable: '+str(set(missing)-available))
            cols = ','.join('"'+c+'"' for c in missing)
            extra = con.execute(f'SELECT game_id,pitcher_id,{cols} FROM model_features').fetchdf()
        a = a.merge(extra, on=['pitcher_id','game_id'], how='left', validate='one_to_one')
    assert source_signature() == before
    return a, events, spec, matches


def clean(frame, features):
    return frame[features].astype(float).replace([np.inf, -np.inf], np.nan)


class PCAEncoder:
    """Median/scale on training rows; rotate selected numeric columns only.

    Missing flags and unrotated columns bypass PCA but retain the same train-only
    scaling as the raw comparator. All-null training features fail explicitly.
    Full-rank rotation changes coordinates, not information or L2 geometry.
    """
    def __init__(self, features, reduced_features=None, components=None):
        self.features = list(features)
        self.reduced_features = list(reduced_features or [])
        self.components = components

    def fit(self, frame):
        x = clean(frame, self.features)
        if x.isna().all().any():
            raise ValueError('All-missing training columns: '+str(x.columns[x.isna().all()].tolist()))
        if not set(self.reduced_features) <= set(self.features):
            raise ValueError('PCA fields must be members of the explicit feature allowlist')
        self.imputer = SimpleImputer(strategy='median', add_indicator=True)
        self.scaler = StandardScaler()
        z = self.scaler.fit_transform(self.imputer.fit_transform(x))
        self.names = self.imputer.get_feature_names_out(self.features).tolist()
        self.rotate_index = [self.names.index(c) for c in self.reduced_features]
        self.keep_index = [j for j in range(z.shape[1]) if j not in self.rotate_index]
        self.pca = None
        if self.rotate_index and self.components != 0:
            k = None if self.components == 'all' else self.components
            with threadpool_limits(limits=1):
                self.pca = PCA(n_components=k, svd_solver='full', whiten=False).fit(z[:, self.rotate_index])
        self.fit_rows = len(frame)
        self.output_dimensions = len(self.keep_index)+(self.pca.n_components_ if self.pca else 0)
        return self

    def standardized(self, frame):
        return self.scaler.transform(self.imputer.transform(clean(frame, self.features)))

    def transform(self, frame):
        z = self.standardized(frame)
        if not self.rotate_index:
            return z
        if self.pca is None:
            return z[:, self.keep_index]
        return np.column_stack([self.pca.transform(z[:, self.rotate_index]), z[:, self.keep_index]])

    def effective_coefficients(self, model):
        """Coefficients in original standardized feature/indicator coordinates."""
        if not self.rotate_index:
            return model.coef_[0].copy()
        if self.pca is None:
            beta = np.zeros(len(self.names))
            beta[self.keep_index] = model.coef_[0]
            return beta
        k = self.pca.n_components_
        beta = np.zeros(len(self.names))
        beta[self.rotate_index] = model.coef_[0, :k] @ self.pca.components_
        beta[self.keep_index] = model.coef_[0, k:]
        return beta


def fit_model(train, targets, features, reduced=None, components=None, tolerance=None):
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        encoder = PCAEncoder(features, reduced, components).fit(train)
        model = LogisticRegression(C=ORIGINAL_CFG['logistic_C'], max_iter=3000,
                                   solver='lbfgs', random_state=CFG['seed'],
                                   tol=CFG['logistic_tolerance'] if tolerance is None else tolerance)
        model.fit(encoder.transform(train), train.target)
        scores = [model.predict_proba(encoder.transform(t))[:, 1] for t in targets]
    return encoder, model, scores


def make_blocks(spec):
    features = spec.loc[spec.selected_for_prediction, 'name'].tolist()
    return {'Current feature space': [c for c in features if c not in BINARY],
            'Prior pitcher profile': list(BASELINE_PAIRS),
            'Within-pitcher departure': list(BASELINE_PAIRS.values())}


def between_pitcher_fraction(values, pitchers):
    """Descriptive between-pitcher SS / total SS; not a fitted random-effect ICC."""
    f = pd.DataFrame({'value': values, 'pitcher': np.asarray(pitchers)})
    means = f.groupby('pitcher').value.transform('mean')
    total = np.sum((f.value-f.value.mean())**2)
    return float(np.sum((means-f.value.mean())**2)/total) if total else np.nan


def subspace_overlap(first, second, k):
    """Mean squared cosines of principal angles; invariant to signs/rotations."""
    singular = np.linalg.svd(first[:k] @ second[:k].T, compute_uv=False)
    return float(np.mean(np.minimum(singular, 1.)**2))


def run_structure(a, spec):
    train = a.loc[a.analysis_eligible & a.partition.eq('train')].copy()
    assert train.game_date.max() <= pd.Timestamp(ORIGINAL_CFG['train_end'])
    blocks = make_blocks(spec)
    definitions = [dict(block=b, feature=c, meaning='strictly prior baseline mean' if b=='Prior pitcher profile'
                       else 'current measurement minus strictly prior pitcher mean, divided by prior SD'
                       if b=='Within-pitcher departure' else 'original selected continuous predictor')
                   for b, cols in blocks.items() for c in cols]
    table(pd.DataFrame(definitions), 'feature_blocks')
    encoders, spectrum, loads, summary, persistence, sensitivities, correlations, missingness = {}, [], [], [], [], [], [], []
    score_frame = a[['row_id','pitcher_id','game_id','game_date','season','partition','analysis_eligible','is_starter','throws_left']].copy()
    for block, columns in blocks.items():
        print(f'Training descriptive PCA: {block}; {len(columns)} measurements; {len(train):,} rows', flush=True)
        encoder = PCAEncoder(columns, columns, 'all').fit(train)
        encoders[block] = encoder
        z = encoder.standardized(train)[:, encoder.rotate_index]
        pca = encoder.pca
        scores = pca.transform(z)
        ratio = pca.explained_variance_ratio_
        prefix = {'Current feature space':'global','Prior pitcher profile':'profile','Within-pitcher departure':'departure'}[block]
        all_scores = pca.transform(encoder.standardized(a)[:, encoder.rotate_index])
        complete = clean(a, columns).notna().all(axis=1)
        score_frame[prefix+'_complete'] = complete.to_numpy()
        for j in range(CFG['plotted_components']):
            # Unit training-PC SD permits comparing magnitudes, never signs across
            # different PCA blocks. This scaling is descriptive only, not predictive.
            score_frame[f'{prefix}_PC{j+1}'] = all_scores[:,j]/np.sqrt(pca.explained_variance_[j])
            persistence.append(dict(block=block, component=j+1,
                between_pitcher_fraction=between_pitcher_fraction(scores[:,j], train.pitcher_id),
                corr_starter=pd.Series(scores[:,j]).corr(train.is_starter.astype(float).reset_index(drop=True)),
                corr_left_handed=pd.Series(scores[:,j]).corr(train.throws_left.astype(float).reset_index(drop=True))))
        corr = np.corrcoef(z, rowvar=False)
        for i in range(len(columns)):
            missingness.append(dict(block=block,feature=columns[i],train_missing_fraction=clean(train,columns)[columns[i]].isna().mean()))
            for j in range(i):
                if abs(corr[i,j]) >= .80:
                    correlations.append(dict(block=block,first=columns[i],second=columns[j],correlation=corr[i,j]))
        for j, variance in enumerate(ratio):
            spectrum.append(dict(block=block,component=j+1,explained_fraction=variance,cumulative_fraction=ratio[:j+1].sum()))
            for i, feature in enumerate(columns):
                loads.append(dict(block=block,component=j+1,feature=feature,weight=pca.components_[j,i],
                    feature_score_correlation=np.corrcoef(z[:,i],scores[:,j])[0,1],
                    squared_weight=pca.components_[j,i]**2))
        summary.append(dict(block=block,features=len(columns),training_rows=len(train),training_pitchers=train.pitcher_id.nunique(),
            first_two_fraction=ratio[:2].sum(),first_three_fraction=ratio[:3].sum(),
            components_80=int(np.searchsorted(np.cumsum(ratio),.8)+1),
            components_90=int(np.searchsorted(np.cumsum(ratio),.9)+1),
            components_95=int(np.searchsorted(np.cumsum(ratio),.95)+1),
            participation_rank=1/np.sum(ratio**2),pairs_abs_corr_90=int(np.sum(np.tril(np.abs(corr)>=.9,k=-1))),
            complete_training_fraction=clean(train,columns).notna().all(axis=1).mean()))
        # All sensitivity bases use the same training-fitted standardization, so
        # subspace overlaps compare coordinates rather than different unit systems.
        rng = np.random.default_rng(CFG['seed'])
        ids = np.concatenate([rng.choice(g, min(len(g),CFG['balanced_training_outings_per_pitcher']), replace=False)
                              for g in train.reset_index(drop=True).groupby('pitcher_id').indices.values()])
        subsets = {'Pitcher capped at 20 outings':ids,
                   'Complete measurements':np.flatnonzero(clean(train,columns).notna().all(axis=1)),
                   'Early training 2015–2018':np.flatnonzero(train.season.le(2018)),
                   'Recent training 2019–2021':np.flatnonzero(train.season.ge(2019))}
        for condition, idx in subsets.items():
            with threadpool_limits(limits=1):
                alternative = PCA(svd_solver='full').fit(z[idx])
            for k in sorted(set([2,3,min(5,len(columns))])):
                sensitivities.append(dict(block=block,condition=condition,rows=len(idx),components=k,
                    subspace_overlap=subspace_overlap(pca.components_,alternative.components_,k)))
    joblib.dump(encoders, CACHE/'descriptive_pca.joblib')
    score_frame.to_parquet(CACHE/'appearance_scores.parquet',index=False)
    for frame,name in [(spectrum,'variance_spectrum'),(loads,'component_loadings'),(summary,'redundancy_summary'),
                       (persistence,'component_persistence'),(sensitivities,'subspace_sensitivity'),
                       (correlations,'high_correlations'),(missingness,'training_missingness')]:
        table(pd.DataFrame(frame),name)
    save_manifest()
    return pd.DataFrame(summary)


def save_manifest():
    result = dict(source_signature=source_signature(), config=CFG, config_sha256=sha(CONFIG_PATH),
        original_cohort_manifest_sha256=sha(ORIGINAL_CACHE/'manifest.json'),
        matching_sha256=sha(ROOT/'reports/injury_type_analysis/tables/matched_controls.csv'),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/injury_analysis').glob('reduction*.py')},
        interpretation='Exploratory reuse of previously examined temporal partitions; not a fresh confirmatory test.')
    (OUT/'manifest.json').write_text(json.dumps(result,indent=2))
    return result

"""Regime labelers (Quiet / Normal / Chaotic) and label sanity checks.

Labels are *hindsight* targets: the labeler's parameters are fitted on the training span only,
but decoding (Viterbi / smoothing) may use the whole sequence. Classifier inputs must be causal;
the labeler's own filtered output is the real-time benchmark.
"""
import json

import numpy as np
import pandas as pd

STATE_NAMES = ('Quiet', 'Normal', 'Chaotic')


def forward_filter(startprob, transmat, means, variances, X):
    """Filtered state probabilities P(s_t | x_1..x_t) of a diagonal Gaussian HMM.

    Standard scaled forward algorithm (Rabiner, 1989). hmmlearn only exposes smoothed
    probabilities P(s_t | x_1..x_T), which use future observations and so cannot run live.
    """
    log_b = -0.5 * (np.log(2 * np.pi * variances)[None] + (X[:, None, :] - means[None]) ** 2 / variances[None]).sum(axis=2)
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    alpha = np.empty_like(b)
    a = startprob * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, len(X)):
        a = (alpha[t - 1] @ transmat) * b[t]
        alpha[t] = a / a.sum()
    return alpha

# Sessions that should be labelled (mostly) Chaotic, used to sanity-check labelers
EVENT_DATES = {
    '2015-08-24': 'China devaluation crash ("Black Monday")',
    '2016-06-24': 'Brexit referendum result',
    '2016-11-09': 'Demonetisation + US election result',
    '2018-09-21': 'IL&FS / DHFL intraday crash',
    '2020-03-12': 'COVID: WHO pandemic declaration sell-off',
    '2020-03-23': 'COVID: lower circuit, lockdown',
    '2021-02-01': 'Union Budget rally',
    '2022-02-24': 'Russia invades Ukraine',
    '2024-06-04': 'General election results',
    '2025-04-07': 'US tariff shock sell-off',
}


def min_duration(labels, k):
    """Merge runs shorter than k bars into the preceding run (removes label flicker)."""
    vals = np.asarray(labels).copy()
    index, name = getattr(labels, 'index', None), getattr(labels, 'name', None)
    if k <= 1 or len(vals) == 0:
        return pd.Series(vals, index=index, name=name)
    change = np.flatnonzero(np.diff(vals)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(vals)]
    for s, e in zip(starts, ends):
        if e - s < k and s > 0:
            vals[s:e] = vals[s - 1]
    return pd.Series(vals, index=index, name=name)


def event_report(labels, events=EVENT_DATES):
    """Share of bars in each regime on each event session."""
    dates = labels.index.tz_localize(None).normalize() if labels.index.tz is not None else labels.index.normalize()
    rows = []
    for d, desc in events.items():
        day = labels[dates == pd.Timestamp(d)]
        if day.empty:
            continue
        shares = day.value_counts(normalize=True).reindex(range(len(STATE_NAMES)), fill_value=0)
        rows.append({'date': d, 'event': desc, **{STATE_NAMES[k]: round(v, 2) for k, v in shares.items()}})
    return pd.DataFrame(rows).set_index('date')


def regime_episodes(labels, state=2):
    """Contiguous runs of `state` as a DataFrame of start, end, n_bars."""
    is_s = (labels == state).to_numpy().astype(int)
    edges = np.diff(np.r_[0, is_s, 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return pd.DataFrame({'start': labels.index[starts], 'end': labels.index[ends - 1], 'n_bars': ends - starts})


class HMMLabeler:
    """3-state Gaussian HMM (hmmlearn) on [deseasonalised return, log deseasonalised range].

    States are ordered by mean log range, so 0 = Quiet, 1 = Normal, 2 = Chaotic.
    sticky: Dirichlet pseudo-count on staying in the same state, as a fraction of the training
    length (biases towards persistent regimes). Fitting / hindsight decoding need hmmlearn; the
    live filter only needs the saved parameters, so the app does not depend on hmmlearn.
    """

    def __init__(self, n_states=3, sticky=0.2, min_bars=4, n_init=4, n_iter=200, tol=1e-3):
        self.n_states, self.sticky, self.min_bars = n_states, sticky, min_bars
        self.n_init, self.n_iter, self.tol = n_init, n_iter, tol

    @staticmethod
    def _raw_obs(f):
        log_range = np.log(f['range_ds'].clip(lower=1e-4))
        return pd.concat([f['z'], log_range], axis=1, keys=['z', 'log_range']).dropna()

    def observations(self, f):
        obs = self._raw_obs(f)
        return obs.index, ((obs - self.mu_) / self.sd_).to_numpy()

    def fit(self, f, start=None, end=None):
        from hmmlearn.hmm import GaussianHMM
        obs = self._raw_obs(f.loc[start:end])
        self.mu_, self.sd_ = obs.mean(), obs.std()
        X = ((obs - self.mu_) / self.sd_).to_numpy()
        prior = 1 + self.sticky * len(X) * np.eye(self.n_states)
        best = None
        for seed in range(self.n_init):
            m = GaussianHMM(self.n_states, covariance_type='diag', n_iter=self.n_iter, tol=self.tol,
                            transmat_prior=prior, random_state=seed).fit(X)
            score = m.score(X)
            if best is None or score > best[0]:
                best = (score, m)
        self.loglik_, m = best
        order = np.argsort(m.means_[:, -1])
        self.startprob_ = m.startprob_[order]
        self.transmat_ = m.transmat_[np.ix_(order, order)]
        self.means_ = m.means_[order]
        self.vars_ = np.diagonal(m.covars_, axis1=1, axis2=2)[order]
        return self

    def _hmmlearn_model(self):
        from hmmlearn.hmm import GaussianHMM
        m = GaussianHMM(self.n_states, covariance_type='diag', init_params='')
        m.n_features = self.means_.shape[1]
        m.startprob_, m.transmat_, m.means_, m.covars_ = self.startprob_, self.transmat_, self.means_, self.vars_
        return m

    def hindsight(self, f):
        """Viterbi labels over the whole frame (uses future bars), then minimum-duration smoothing."""
        idx, X = self.observations(f)
        path = self._hmmlearn_model().predict(X)
        return min_duration(pd.Series(path, index=idx, name='regime'), self.min_bars)

    def smoothed_proba(self, f):
        idx, X = self.observations(f)
        return pd.DataFrame(self._hmmlearn_model().predict_proba(X), index=idx, columns=STATE_NAMES)

    def filtered_proba(self, f):
        """Real-time state probabilities (forward filter only)."""
        idx, X = self.observations(f)
        p = forward_filter(self.startprob_, self.transmat_, self.means_, self.vars_, X)
        return pd.DataFrame(p, index=idx, columns=STATE_NAMES)

    def live(self, f):
        p = self.filtered_proba(f)
        return pd.Series(p.to_numpy().argmax(axis=1), index=p.index, name='regime')

    def summary(self):
        means = self.means_ * self.sd_.to_numpy() + self.mu_.to_numpy()
        sds = np.sqrt(self.vars_) * self.sd_.to_numpy()
        return pd.DataFrame({'z_sd': sds[:, 0], 'range_ds_typical': np.exp(means[:, 1]),
                             'expected_duration_bars': 1 / (1 - np.diag(self.transmat_))},
                            index=STATE_NAMES[:self.n_states])

    _PARAMS = ('startprob_', 'transmat_', 'means_', 'vars_')

    def to_json(self):
        return json.dumps({**{k: getattr(self, k).tolist() for k in self._PARAMS},
                           'mu': self.mu_.to_dict(), 'sd': self.sd_.to_dict(), 'min_bars': self.min_bars})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        lab = cls(min_bars=d['min_bars'])
        for k in cls._PARAMS:
            setattr(lab, k, np.asarray(d[k]))
        lab.mu_, lab.sd_ = pd.Series(d['mu']), pd.Series(d['sd'])
        return lab


class GMMLabeler:
    """Gaussian mixture on the same observations as HMMLabeler, but with no temporal model.

    Live labels are the per-bar cluster; hindsight labels add minimum-duration smoothing.
    """

    def __init__(self, n_states=3, min_bars=4, random_state=0):
        from sklearn.mixture import GaussianMixture
        self.n_states, self.min_bars = n_states, min_bars
        self.gmm = GaussianMixture(n_states, covariance_type='diag', n_init=4, random_state=random_state)

    _raw_obs = staticmethod(HMMLabeler._raw_obs)

    def _obs(self, f):
        obs = self._raw_obs(f)
        return obs.index, ((obs - self.mu_) / self.sd_).to_numpy()

    def fit(self, f, start=None, end=None):
        obs = self._raw_obs(f.loc[start:end])
        self.mu_, self.sd_ = obs.mean(), obs.std()
        self.gmm.fit(((obs - self.mu_) / self.sd_).to_numpy())
        self.order_ = np.argsort(np.argsort(self.gmm.means_[:, -1]))  # old cluster -> ordered state
        return self

    def live(self, f):
        idx, X = self._obs(f)
        return pd.Series(self.order_[self.gmm.predict(X)], index=idx, name='regime')

    def hindsight(self, f):
        return min_duration(self.live(f), self.min_bars).rename('regime')


class GARCHLabeler:
    """GARCH(1,1)-t on deseasonalised returns; conditional volatility binned by training percentiles.

    The conditional volatility is a one-step-ahead (causal) quantity, so live labels are the raw
    bins and hindsight labels add minimum-duration smoothing.
    """

    def __init__(self, quantiles=(0.30, 0.94), min_bars=4):
        self.quantiles, self.min_bars = quantiles, min_bars

    def _model(self, f):
        from arch import arch_model
        z = f['z'].dropna() * 1000  # rescale for optimiser stability
        return arch_model(z, mean='Zero', vol='GARCH', p=1, q=1, dist='t', rescale=False)

    def fit(self, f, start=None, end=None):
        self.params_ = self._model(f.loc[start:end]).fit(disp='off').params
        cv = self.conditional_vol(f.loc[start:end])
        self.thresholds_ = cv.quantile(list(self.quantiles)).to_numpy()
        return self

    def conditional_vol(self, f):
        return self._model(f).fix(self.params_).conditional_volatility

    def live(self, f):
        cv = self.conditional_vol(f)
        return pd.Series(np.searchsorted(self.thresholds_, cv.to_numpy()), index=cv.index, name='regime')

    def hindsight(self, f):
        return min_duration(self.live(f), self.min_bars).rename('regime')

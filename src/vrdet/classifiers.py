"""Real-time regime classifiers: each maps the features available at the close of bar t to the regime of bar t.

All inputs are causal (built by `features.build_features`); the targets are the HMM hindsight labels.
"""
import numpy as np
import pandas as pd


class ThresholdClassifier:
    """Single-feature rule: bin one volatility estimate at training cut-offs that reproduce the label shares.

    The simplest defensible baseline (e.g. "2-hour Garman-Klass volatility above its 93rd percentile = Chaotic").
    """

    def __init__(self, col='gk_8'):
        self.col = col

    def fit(self, X, y):
        shares = np.bincount(np.asarray(y), minlength=3) / len(y)
        self.cuts_ = X[self.col].quantile(np.cumsum(shares)[:-1]).to_numpy()
        return self

    def predict(self, X):
        x = X[self.col].to_numpy()
        return np.where(np.isnan(x), 1, np.searchsorted(self.cuts_, x))


def logistic_regression(C=1.0):
    """Multinomial logistic regression on standardised features (missing values -> training median)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def lightgbm(**overrides):
    """Gradient-boosted trees; handles missing values natively."""
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=0, verbose=-1)
    return LGBMClassifier(**{**params, **overrides})


class TunedChaoticThreshold:
    """Calls Chaotic when P(Chaotic) >= tau, otherwise the more likely of Quiet / Normal.

    Chaotic is rare (~5-7% of bars), so the default argmax rule is conservative. tau is chosen
    without touching the test data: fit on the training window minus its last year, pick the tau that
    maximises Chaotic F1 on that last year, then refit on the whole training window.
    """

    def __init__(self, make, grid=np.round(np.arange(0.10, 0.65, 0.05), 2)):
        self.make, self.grid = make, grid

    def fit(self, X, y):
        from sklearn.metrics import f1_score
        y = np.asarray(y)
        val = X.index.year == X.index.year.max()
        p = self.make().fit(X[~val], y[~val]).predict_proba(X[val])[:, 2]
        scores = [f1_score(y[val] == 2, p >= t) for t in self.grid]
        self.tau_ = float(self.grid[int(np.argmax(scores))])
        self.model_ = self.make().fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.where(p[:, 2] >= self.tau_, 2, p[:, :2].argmax(axis=1))


def garch_live(f, train_start, train_end, y_train):
    """GARCH(1,1)-t conditional volatility binned at training cut-offs that reproduce the label shares."""
    from .labels import GARCHLabeler
    shares = np.bincount(np.asarray(y_train), minlength=3) / len(y_train)
    lab = GARCHLabeler(quantiles=tuple(np.cumsum(shares)[:-1])).fit(f, start=train_start, end=train_end)
    return lab.live(f)


def walk_forward(models, index, test_years, train_start):
    """Expanding-window evaluation: for each test year, train on [train_start, year) and predict that year.

    models: {name: fit_predict(train_mask, test_mask, train_end) -> predictions for the test rows}
    Returns {name: Series of out-of-sample predictions (or probabilities) over all test years}.
    """
    out = {}
    for name, fit_predict in models.items():
        parts = []
        for year in test_years:
            start = pd.Timestamp(f'{year}-01-01', tz=index.tz)
            train = (index >= pd.Timestamp(train_start, tz=index.tz)) & (index < start)
            test = index.year == year
            pred = fit_predict(train, test, f'{year - 1}-12-31')
            parts.append(pd.Series(np.asarray(pred), index=index[test]))
        out[name] = pd.concat(parts).rename(name)
    return out

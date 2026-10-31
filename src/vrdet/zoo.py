"""Model zoo: every real-time classifier compared in Stage 04 and playable in the dashboard's Model Lab.

All models see the same inputs as the production model (causal features + HMM filter probabilities),
are wrapped in `TunedChaoticThreshold` (Chaotic when P(Chaotic) >= tau, tau tuned on the last training
year) and are evaluated with the same walk-forward protocol (scripts/train_zoo.py).

Every estimator here has fit(X, y) / predict_proba(X) on DataFrames. Sequence models build their own
windows from consecutive rows, so callers pass `CONTEXT` rows of history before the rows they need.
"""
import numpy as np

from .classifiers import lightgbm, logistic_regression

CONTEXT = 32   # rows of history passed before the rows to predict (>= the longest sequence window)
N_JOBS = 8     # leave half of the 16 threads free for the live runner / dashboard


def _impute():
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy='median', keep_empty_features=True)


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    return make_pipeline(_impute(), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=50, max_features='sqrt', class_weight='balanced_subsample',
        n_jobs=N_JOBS, random_state=0))


def extra_trees():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline
    return make_pipeline(_impute(), ExtraTreesClassifier(
        n_estimators=300, min_samples_leaf=50, max_features='sqrt', class_weight='balanced_subsample',
        n_jobs=N_JOBS, random_state=0))


def adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    return make_pipeline(_impute(), AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3, min_samples_leaf=100), n_estimators=100, learning_rate=0.5,
        random_state=0))


def xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, reg_lambda=1.0, tree_method='hist', n_jobs=N_JOBS, random_state=0)


def mlp():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(_impute(), StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=60,
        early_stopping=True, n_iter_no_change=6, random_state=0))


def lstm():
    return SequenceClassifier('lstm')


def gru():
    return SequenceClassifier('gru')


def transformer():
    return SequenceClassifier('transformer', lr=1e-3)


def tcn():
    return SequenceClassifier('tcn')


def hist_gb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.08, l2_leaf_reg=3.0,
                              loss_function='MultiClass', thread_count=N_JOBS, random_seed=0, verbose=0,
                              allow_writing_files=False)   # no catboost_info/ log folder in the working directory


def decision_tree():
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    return make_pipeline(_impute(), DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=0))


def knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(_impute(), StandardScaler(), KNeighborsClassifier(n_neighbors=100, n_jobs=N_JOBS))


def naive_bayes():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    # heavy-tailed volatility inputs -> normal scores, closer to the Gaussian class-conditional assumption
    return make_pipeline(_impute(), QuantileTransformer(output_distribution='normal', random_state=0), GaussianNB())


def lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(_impute(), StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


def qda():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # many inputs are near-collinear (overlapping windows), so the class covariances need shrinkage
    return make_pipeline(_impute(), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.2))


def svm():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    # exact RBF-kernel SVC is O(n^2) on 60k bars; a 500-landmark Nystroem map + linear SVM approximates it.
    # SVM scores are not probabilities, so they are calibrated (sigmoid, 3 folds).
    return make_pipeline(_impute(), StandardScaler(),
                         Nystroem(kernel='rbf', gamma=1 / 112, n_components=500, random_state=0),
                         CalibratedClassifierCV(LinearSVC(C=0.1), method='sigmoid', cv=3))


def ordinal():
    return Ordinal()


def soft_vote():
    return SoftVote(['lightgbm', 'xgboost', 'random_forest'])


def stacking():
    return Stacking(['lightgbm', 'xgboost', 'logistic'])


MAKERS = {'lightgbm': lightgbm, 'logistic': logistic_regression, 'random_forest': random_forest,
          'extra_trees': extra_trees, 'adaboost': adaboost, 'xgboost': xgboost, 'mlp': mlp,
          'lstm': lstm, 'gru': gru, 'transformer': transformer, 'soft_vote': soft_vote, 'stacking': stacking,
          'catboost': catboost, 'hist_gb': hist_gb, 'decision_tree': decision_tree, 'knn': knn,
          'naive_bayes': naive_bayes, 'lda': lda, 'qda': qda, 'svm': svm, 'tcn': tcn, 'ordinal': ordinal}

# key -> display name, family, one-line description (shown in the Model Lab)
MODELS = {
    'lightgbm': ('LightGBM + HMM', 'Gradient boosting', 'Production model. Leaf-wise boosted trees; handles missing values natively.'),
    'logistic': ('Logistic regression', 'Linear', 'Multinomial logistic regression on standardised inputs: the interpretable baseline.'),
    'random_forest': ('Random Forest', 'Bagging', '300 deep-ish trees on bootstrap samples, class-balanced; averages away variance.'),
    'extra_trees': ('Extra Trees', 'Bagging', 'Like Random Forest but with random split points: smoother, faster, more biased.'),
    'adaboost': ('AdaBoost', 'Boosting', '100 shallow trees, each focusing on the bars the previous ones got wrong.'),
    'xgboost': ('XGBoost', 'Gradient boosting', 'Depth-wise boosted trees with second-order gradients and regularisation.'),
    'mlp': ('MLP', 'Neural network', 'Feed-forward network (64-32) on the current bar only; early stopping.'),
    'lstm': ('LSTM', 'Sequence model', 'Long short-term memory network over the last 16 bars (4 hours) of inputs.'),
    'gru': ('GRU', 'Sequence model', 'Gated recurrent unit network over the last 16 bars (4 hours) of inputs.'),
    'transformer': ('Transformer', 'Sequence model', 'Self-attention encoder (2 layers, 4 heads) over the last 16 bars (4 hours) of inputs.'),
    'soft_vote': ('Soft voting', 'Ensemble', 'Average of LightGBM, XGBoost and Random Forest probabilities.'),
    'stacking': ('Stacking', 'Ensemble', 'Logistic meta-model on LightGBM, XGBoost and logistic probabilities, fitted on a held-out year.'),
    'catboost': ('CatBoost', 'Gradient boosting', 'Symmetric (oblivious) trees with ordered boosting, which reduces target leakage.'),
    'hist_gb': ('Hist. gradient boosting', 'Gradient boosting', 'scikit-learn histogram boosting (LightGBM-style binning), native missing values.'),
    'decision_tree': ('Decision tree', 'Tree', 'One tree, depth 6: readable if/then rules, the single-tree baseline for the ensembles.'),
    'knn': ('k-NN', 'Instance-based', 'Regime shares among the 100 most similar past bars (standardised inputs); no training step.'),
    'naive_bayes': ('Naive Bayes', 'Probabilistic', 'Gaussian naive Bayes on normal-score inputs: assumes inputs are independent given the regime.'),
    'lda': ('LDA', 'Discriminant analysis', 'Linear discriminant analysis with shrinkage: one shared covariance, linear boundaries.'),
    'qda': ('QDA', 'Discriminant analysis', 'Quadratic discriminant analysis: one covariance per regime, curved boundaries (regularised).'),
    'svm': ('SVM (RBF)', 'Kernel method', 'Support vector machine with an RBF kernel (500-landmark Nystroem approximation), calibrated.'),
    'tcn': ('Temporal CNN', 'Sequence model', 'Dilated causal 1-D convolutions (dilations 1, 2, 4) over the last 16 bars of inputs.'),
    'ordinal': ('Ordinal LightGBM', 'Ordinal', 'Uses the order Quiet < Normal < Chaotic: LightGBMs for P(above Quiet) and P(above Normal).'),
}


def _holdout(index):
    """Last calendar year of a training index: the validation slice used by stacking (and tau tuning)."""
    return index.year == index.year.max()


class SoftVote:
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.models_ = [MAKERS[k]().fit(X, y) for k in self.members]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


class Stacking:
    """Blending: base models fitted without the last training year predict it; a logistic meta-model learns
    from those out-of-sample log-probabilities; then the bases are refitted on all training data."""

    def __init__(self, members):
        self.members = members

    def _meta_inputs(self, models, X):
        return np.hstack([np.log(np.clip(m.predict_proba(X), 1e-6, 1)) for m in models])

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y)
        val = _holdout(X.index)
        bases = [MAKERS[k]().fit(X[~val], y[~val]) for k in self.members]
        self.meta_ = LogisticRegression(C=1.0, max_iter=2000).fit(self._meta_inputs(bases, X[val]), y[val])
        self.models_ = [MAKERS[k]().fit(X, y) for k in self.members]
        return self

    def predict_proba(self, X):
        return self.meta_.predict_proba(self._meta_inputs(self.models_, X))


class Ordinal:
    """Frank & Hall ordinal classification: binary models for P(y > 0) and P(y > 1), combined into 3 classes."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.models_ = [lightgbm().fit(X, (y > k).astype(int)) for k in (0, 1)]
        return self

    def predict_proba(self, X):
        above = [m.predict_proba(X)[:, 1] for m in self.models_]
        chaotic = np.minimum(above[1], above[0])          # keep the cumulative probabilities ordered
        p = np.column_stack([1 - above[0], above[0] - chaotic, chaotic])
        return p / p.sum(axis=1, keepdims=True)


class SequenceClassifier:
    """LSTM / GRU / Transformer / temporal CNN over the last `window` rows. Inputs: median-imputed, standardised features.

    Windows are built from consecutive rows of the frame passed in (the first rows are zero-padded),
    so pass CONTEXT rows of history before the rows of interest.
    """

    def __init__(self, cell='lstm', window=16, hidden=48, epochs=8, lr=2e-3, batch=256, dropout=0.2,
                 seed=0, threads=N_JOBS):
        self.cell, self.window, self.hidden, self.epochs = cell, window, hidden, epochs
        self.lr, self.batch, self.dropout, self.seed, self.threads = lr, batch, dropout, seed, threads

    def _prep(self, X):
        import torch
        A = ((X.fillna(self.med_) - self.mu_) / self.sd_).fillna(0).to_numpy(np.float32)
        pad = np.zeros((self.window - 1, A.shape[1]), np.float32)
        return torch.from_numpy(np.vstack([pad, A]))

    def _batches(self, A, rows):
        import torch
        offsets = torch.arange(self.window)
        for i in range(0, len(rows), self.batch):
            b = rows[i:i + self.batch]
            yield b, A[b[:, None] + offsets]   # row r of the frame is padded position r + window - 1

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        self.med_ = X.median()
        filled = X.fillna(self.med_)
        self.mu_, self.sd_ = filled.mean(), filled.std().replace(0, 1)
        A, target = self._prep(X), torch.as_tensor(np.array(y), dtype=torch.long)
        from .seqnet import ConvNet, RecurrentNet, TransformerNet
        self.net_ = (TransformerNet(X.shape[1], self.window, self.hidden, self.dropout) if self.cell == 'transformer'
                     else ConvNet(X.shape[1], self.hidden, self.dropout) if self.cell == 'tcn'
                     else RecurrentNet(self.cell, X.shape[1], self.hidden, self.dropout))
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = torch.nn.CrossEntropyLoss()
        gen = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            for b, xb in self._batches(A, torch.randperm(len(X), generator=gen)):
                opt.zero_grad()
                loss = loss_fn(self.net_(xb), target[b])
                loss.backward()
                opt.step()
        self.net_.eval()
        return self

    def predict_proba(self, X):
        import torch
        torch.set_num_threads(self.threads)
        A, out = self._prep(X), []
        with torch.no_grad():
            for _, xb in self._batches(A, torch.arange(len(X))):
                out.append(torch.softmax(self.net_(xb), dim=1).numpy())
        return np.vstack(out)

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON, MAX_INT, _prepare_fit_binary, _get_plain_sgd_function
from sklearn.base import clone
from sklearn.utils import check_random_state, compute_class_weight, deprecated
from sklearn.utils.parallel import Parallel, delayed
from sklearn.linear_model._base import make_dataset


class SGDClassifierFixedSplit(SGDClassifier):
    def __init__(
        self,
        validation_mask,
        loss="hinge",
        *,
        penalty="l2",
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        epsilon=DEFAULT_EPSILON,
        n_jobs=None,
        random_state=None,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
    ):
        self.validation_mask=validation_mask
        super().__init__(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            shuffle=shuffle,
            verbose=verbose,
            epsilon=epsilon,
            n_jobs=n_jobs,
            random_state=random_state,
            learning_rate=learning_rate,
            eta0=eta0,
            power_t=power_t,
            early_stopping=early_stopping,
            validation_fraction=0,
            n_iter_no_change=n_iter_no_change,
            class_weight=class_weight,
            warm_start=warm_start,
            average=average,
        )

    def _make_validation_split(self, y, sample_mask):
        return self.validation_mask

    def _make_validation_score_cb(
            self, validation_mask, X, y, sample_weight, classes=None
        ):
            if not self.early_stopping:
                return None

            return _ValidationScoreCallback(
                self,
                X[validation_mask],
                y[validation_mask],
                sample_weight[validation_mask],
                classes=classes,
            )




    def _fit_multiclass(self, X, y, alpha, C, learning_rate, sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).
        """
        # Precompute the validation split using the multiclass labels
        # to ensure proper balancing of the classes.
        validation_mask = self._make_validation_split(y, sample_mask=sample_weight > 0)

        # Use joblib to fit OvA in parallel.
        # Pick the random seed for each job outside of fit_binary to avoid
        # sharing the estimator random state between threads which could lead
        # to non-deterministic behavior
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        # result = Parallel(
        #     n_jobs=self.n_jobs, verbose=self.verbose, require="sharedmem"
        # )(
        #     delayed(fit_binary)(
        #         self,
        #         i,
        #         X,
        #         y,
        #         alpha,
        #         C,
        #         learning_rate,
        #         max_iter,
        #         self._expanded_class_weight[i],
        #         1.0,
        #         sample_weight,
        #         validation_mask=validation_mask,
        #         random_state=seed,
        #     )
        #     for i, seed in enumerate(seeds)
        # )
        result = [fit_binary(
                self,
                i,
                X,
                y,
                alpha,
                C,
                learning_rate,
                max_iter,
                self._expanded_class_weight[i],
                1.0,
                sample_weight,
                validation_mask=validation_mask,
                random_state=seed,
            )
            for i, seed in enumerate(seeds)
         ]



        # take the maximum of n_iter_ over every binary fit
        n_iter_ = 0.0
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self._average_coef
                self.intercept_ = self._average_intercept
            else:
                self.coef_ = self._standard_coef
                self._standard_intercept = np.atleast_1d(self.intercept_)
                self.intercept_ = self._standard_intercept


def fit_binary(
    est,
    i,
    X,
    y,
    alpha,
    C,
    learning_rate,
    max_iter,
    pos_weight,
    neg_weight,
    sample_weight,
    validation_mask=None,
    random_state=None,
):
    """Fit a single binary classifier.

    The i'th class is considered the "positive" class.

    Parameters
    ----------
    est : Estimator object
        The estimator to fit

    i : int
        Index of the positive class

    X : numpy array or sparse matrix of shape [n_samples,n_features]
        Training data

    y : numpy array of shape [n_samples, ]
        Target values

    alpha : float
        The regularization parameter

    C : float
        Maximum step size for passive aggressive

    learning_rate : str
        The learning rate. Accepted values are 'constant', 'optimal',
        'invscaling', 'pa1' and 'pa2'.

    max_iter : int
        The maximum number of iterations (epochs)

    pos_weight : float
        The weight of the positive class

    neg_weight : float
        The weight of the negative class

    sample_weight : numpy array of shape [n_samples, ]
        The weight of each sample

    validation_mask : numpy array of shape [n_samples, ], default=None
        Precomputed validation mask in case _fit_binary is called in the
        context of a one-vs-rest reduction.

    random_state : int, RandomState instance, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    # if average is not true, average_coef, and average_intercept will be
    # unused
    y_i, coef, intercept, average_coef, average_intercept = _prepare_fit_binary(
        est, y, i, input_dtye=X.dtype
    )
    assert y_i.shape[0] == y.shape[0] == sample_weight.shape[0]

    random_state = check_random_state(random_state)
    dataset, intercept_decay = make_dataset(
        X, y_i, sample_weight, random_state=random_state
    )

    penalty_type = est._get_penalty_type(est.penalty)
    learning_rate_type = est._get_learning_rate_type(learning_rate)

    if validation_mask is None:
        validation_mask = est._make_validation_split(y_i, sample_mask=sample_weight > 0)
    classes = np.array([-1, 1], dtype=y_i.dtype)
    validation_score_cb = est._make_validation_score_cb(
        validation_mask, X, y_i, sample_weight, classes=classes
    )

    # numpy mtrand expects a C long which is a signed 32 bit integer under
    # Windows
    seed = random_state.randint(MAX_INT)

    tol = est.tol if est.tol is not None else -np.inf

    _plain_sgd = _get_plain_sgd_function(input_dtype=coef.dtype)
    coef, intercept_out, average_coef, average_intercept, n_iter_ = _plain_sgd(
        coef,
        intercept,
        average_coef,
        average_intercept,
        est._loss_function_,
        penalty_type,
        alpha,
        C,
        est.l1_ratio,
        dataset,
        validation_mask,
        est.early_stopping,
        validation_score_cb,
        int(est.n_iter_no_change),
        max_iter,
        tol,
        int(est.fit_intercept),
        int(est.verbose),
        int(est.shuffle),
        seed,
        pos_weight,
        neg_weight,
        learning_rate_type,
        est.eta0,
        est.power_t,
        0,
        est.t_,
        intercept_decay,
        est.average,
    )
    if est.verbose>=2:
        print(f"best epoch was epoch {validation_score_cb.best_call} of {n_iter_} epochs. Best score was {validation_score_cb.best_score}")

        #print(validation_score_cb(coef, intercept_out), validation_score_cb(validation_score_cb.best_coef_, validation_score_cb.best_intercept_))
        #print(validation_score_cb.best_intercept_)

    coef[:]=validation_score_cb.best_coef_[:]
    intercept=validation_score_cb.best_intercept_[0]

    if est.average:
        if len(est.classes_) == 2:
            est._average_intercept[0] = average_intercept
        else:
            est._average_intercept[i] = average_intercept

    return coef, intercept, n_iter_



class _ValidationScoreCallback:
    """Callback for early stopping based on validation score"""

    def __init__(self, estimator, X_val, y_val, sample_weight_val, classes=None, verbose=False):
        self.estimator = clone(estimator)
        self.estimator.t_ = 1  # to pass check_is_fitted
        if classes is not None:
            self.estimator.classes_ = classes
        self.X_val = X_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val
        self.best_score=-1
        self.best_coef_ = None
        self.best_intercept_ = None
        self.verbose=verbose
        self.num_calls = 0
        self.best_call = -1

    def __call__(self, coef, intercept):
        est = self.estimator
        #print(type(est.coef_))
        est.coef_ = coef.reshape(1, -1)
        est.intercept_ = np.atleast_1d(intercept)
        score =  est.score(self.X_val, self.y_val, self.sample_weight_val)
        if score > self.best_score:
            if self.verbose>=3:
                print("new_best_score", score)
            self.best_score=score
            self.best_call=self.num_calls
            if self.best_coef_ is None:
                self.best_coef_ = coef.reshape(1, -1).copy()
            else:
                self.best_coef_[:] = coef.reshape(1, -1)
            if self.best_intercept_ is None:
                self.best_intercept_ =  np.atleast_1d(intercept).copy()
            else:
                self.best_intercept_[:] =  np.atleast_1d(intercept)
        self.num_calls+=1
        return score
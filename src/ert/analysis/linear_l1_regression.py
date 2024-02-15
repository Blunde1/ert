import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def linear_l1_regression(D, X, max_iter=100, cv_folds=5):
    """
    Performs LASSO regression for each response in X against predictors in D,
    constructing a sparse matrix of regression coefficients.

    The function scales features in D using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of D. This
    extracts the effect of each feature in D on each response in X, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    D : np.ndarray
        2D array of predictors with shape (n, p).
    X : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H : np.ndarray
        2D array of responses with shape (m, p) with re-scaled LASSO
        regression coefficients for each response in X.

    Raises
    ------
    AssertionError
        If the number of samples in D and X do not match, or if the shape of
        H is not (m, p).
    """
    n, p = D.shape  # p: number of features
    n_y, m = X.shape  # m: number of y responses

    # Assert that the first dimension of D and X are the same
    assert n == n_y, "Number of samples in D and X must be the same"

    scaler_d = StandardScaler()
    D_scaled = scaler_d.fit_transform(D)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # Loop over features
    H = np.zeros((m, p))
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        x_j = X_scaled[:, j]

        # Learn individual regularization and fit
        eps = 1e-3
        model_cv = LassoCV(cv=cv_folds, fit_intercept=False, max_iter=max_iter, eps=eps)
        model_cv.fit(D_scaled, x_j)

        # Extract coefficients
        for non_zero_ind in model_cv.coef_.nonzero()[0]:
            H[j, non_zero_ind] = (
                scaler_x.scale_[j]
                * model_cv.coef_[non_zero_ind]
                / scaler_d.scale_[non_zero_ind]
            )

    # Assert shape of H_sparse
    assert H.shape == (m, p), "Shape of H_sparse must be (m, p)"

    return H

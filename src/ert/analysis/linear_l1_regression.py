import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2
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


def expected_max_chisq(p):
    """Expected maximum of p central chi-square(1) random variables"""

    def dmaxchisq(x):
        return 1.0 - np.exp(p * chi2.logcdf(x, df=1))

    expectation, _ = quad(dmaxchisq, 0, np.inf)
    return expectation


def mse(residuals):
    return 0.5 * np.mean(residuals**2)


def calculate_psi_M(x, y, beta_estimate):
    """The psi/score function for mse: 0.5*residual**2"""
    residuals = y - beta_estimate * x
    psi = -residuals * x
    M = -np.mean(x**2)
    return psi, M


def calculate_influence(x, y, beta_estimate):
    """The influence of (x, y) on beta_estimate as an mse M-estimator"""
    psi, M = calculate_psi_M(x, y, beta_estimate)
    return psi / M


def boost_linear_regression(X, y, learning_rate=0.3, tol=1e-6, max_iter=10000):
    """Boost coefficients of linearly regressing y on standardized X.

    The coefficient selection utilizes information theoretic weighting.
    The stopping criterion utilizes information theoretic loss-reduction.
    """
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    residuals, residuals_loo = y.copy(), y.copy()
    y_mse = mse(y)
    previous_residual_mse = y_mse
    mse_factor = expected_max_chisq(n_features)
    comparison_factor_aic = 1 + mse_factor / n_samples
    # The (fast) AIC comparison factor assumes the AIC on mse as n_features/n_samples
    # A stricter criterion is the loo-adjustment: mse(residuals_loo)-mse(residuals). This converges to TIC. Under certain conditions this is AIC.
    # At worst, we are maximizing squares. See Lunde 2020 Appendix A. This needs to be adjusted for.
    # The mse_factor adjusts for this.

    for iteration in range(max_iter):
        coef_changes = np.zeros(n_features)

        # Calculate coefficient change for each feature
        for feature in range(n_features):
            coef_changes[feature] = np.dot(X[:, feature], residuals) / n_samples

        # adjust based on information-criterion penalty
        feature_evaluation = np.zeros(n_features)
        for feature in range(n_features):
            beta_estimate_j = coef_changes[feature]
            residuals_j = residuals - beta_estimate_j * X[:, feature]
            residual_mse_j = mse(residuals_j)  # np.cov(residuals_j, rowvar=False)
            if coefficients[feature] == 0:
                # add IC penalty: for mse, this is 1 x conditional variance (aic (fast) context), because feature is not added
                # The added feature IC penalty is constant for both models, and therefore not added.
                feature_evaluation[feature] = residual_mse_j * comparison_factor_aic
            else:
                feature_evaluation[feature] = residual_mse_j

        # Select feature based on loss criterion
        best_feature = np.argmin(feature_evaluation)
        beta_estimate = coef_changes[best_feature]
        coef_change = beta_estimate * learning_rate

        # adjust to loo estimates for coef_change
        influence = calculate_influence(X[:, best_feature], residuals, beta_estimate)
        beta_estimate_loo = beta_estimate - influence / n_samples
        coef_change_loo = beta_estimate_loo * learning_rate

        # Make sure boosting should start at all
        if iteration == 0:
            residuals_full = y - beta_estimate * X[:, best_feature]
            residuals_full_loo = y - beta_estimate_loo * X[:, best_feature]
            if y_mse < mse(residuals_full) + mse_factor * (
                mse(residuals_full_loo) - mse(residuals_full)
            ):
                break

        # Update residuals
        residuals -= coef_change * X[:, best_feature]
        residuals_loo -= coef_change_loo * X[:, best_feature]

        # Do loo-cv-square-maximized adjusted mse residual estimate
        new_residual_mse = mse(residuals) + mse_factor * (
            mse(residuals_loo) - mse(residuals)
        )

        # Check for convergence
        if np.abs(coef_change) < tol or previous_residual_mse < new_residual_mse:
            break
        else:
            # Update
            previous_residual_mse = new_residual_mse
            coefficients[best_feature] += coef_change

    return coefficients


def linear_boost_ic_regression(D, X):
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

        # Learn individual fit
        coefficients_j = boost_linear_regression(D_scaled, x_j)

        # Extract coefficients
        for non_zero_ind in coefficients_j.nonzero()[0]:
            H[j, non_zero_ind] = (
                scaler_x.scale_[j]
                * coefficients_j[non_zero_ind]
                / scaler_d.scale_[non_zero_ind]
            )

    # Assert shape of H_sparse
    assert H.shape == (m, p), "Shape of H_sparse must be (m, p)"

    return H

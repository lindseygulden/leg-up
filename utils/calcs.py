"""functions to calculate stuff."""

from typing import List, Tuple, Union

import numpy as np
from scipy.linalg import svd
from statsmodels.stats.contingency_tables import Table2x2


def odds_ratio(
    condition1_yes_no: Union[Tuple[int], List[int]],
    condition2_yes_no: Union[Tuple[int], List[int]],
    ci: float = 0.95,
):
    """Computes the odds ratio and the upper and lower bounds of the confidence interval
    Args:
        condition1_yes_no: with condition 1 [# with event, # without event]
        condition2_yes_no: with condition 2 [# with event, # without event]
        ci: float b/w 0 and 1 (confidence interval) = (1- alpha)
    Returns:
        tuple: odds ratio for data, lower bound, upper bound (all floats)
    """
    if (ci < 0) | (ci > 1):
        raise ValueError("Confidence interval value must be between 0 and 1")
    if (
        (not isinstance(condition1_yes_no, list))
        | (not isinstance(condition2_yes_no, list))
        | (len(condition1_yes_no) != 2)
        | (len(condition2_yes_no) != 2)
    ):
        raise TypeError(
            "Arguments condition1_yes_no and condition2_yes_no should be two-member lists of integer values"
        )

    t = Table2x2([condition1_yes_no, condition2_yes_no])
    upper_lower = t.oddsratio_confint((1 - ci))
    return t.oddsratio, upper_lower[0], upper_lower[1]


def scale_zero_to_one(x: float, minval: float, maxval: float):
    """Scales input argument x along the range [minval,maxval] to [0,1]
    Args:
        x: float to be scaled to [0,1]
        minval: float that defines lower end of scaling range
        maxval: float that defines upper end of scaling range
    Returns:
        scaled version of x
    """

    if (not isinstance(x, float)) & (not isinstance(x, int)):
        raise TypeError(f"x is {x}: x must be numeric (float or int)")
    if (not isinstance(minval, float)) & (not isinstance(minval, int)):
        raise TypeError(f"x is {x}: minval must be numeric (float or int)")
    if (not isinstance(maxval, float)) & (not isinstance(maxval, int)):
        raise TypeError(f"x is {x}: maxval must be numeric (float or int)")
    if maxval <= minval:
        raise ValueError("maxval must be bigger than minval")
    if (x < minval) | (x > maxval):
        raise ValueError(
            f"x ({x}) must be >= minval ({minval}) and <=maxval ({maxval})"
        )
    return (x - minval) / (maxval - minval)


def total_least_squares_regression(X: np.array, y: np.array, intercept: bool = True):
    """
    Implements TLS/Demming regression with optional intercept using SVD
    Args:
        X: An input data matrix, X, with dimensions (n_samples, n_features)
        y: An input vector for target (y) with dimension (n_samples,)
    Returns:
        b: float intercept
        coef: an array with dimensions (n_features,), estimating coefficient values
    """
    n_samples = X.shape[0]

    # make y a column vector
    y = y.reshape(-1, 1)

    # Make sure X is an array:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if intercept:
        x_aug = np.hstack((np.ones((n_samples, 1)), X))
    else:
        x_aug = X

    z_matrix = np.hstack((x_aug, y))

    _, _, vt = svd(z_matrix, full_matrices=False)

    # extract the smallest right singular vector to get the regression coefficients
    v = vt[-1]

    v_x = v[:-1]  # intercept and coefficients
    v_y = v[-1]

    if np.isclose(v_y, 0):
        raise ValueError("TLS solution undefined (v_y ≈ 0)")

    # TLS parameters
    params = -v_x / v_y
    intercept = params[0]
    coef = params[1:]

    return intercept, list(coef)

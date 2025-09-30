import numpy as np

def logit_to_prob(mod, x):
    """
    Compute probabilities from a model’s raw predictions using the logistic (sigmoid) function.

    Parameters
    ----------
    mod : object
        A model object that implements a `predict` method (e.g., scikit-learn or statsmodels model).
        The `predict` method should return raw (untransformed) prediction values.
    x : array-like or DataFrame
        Input features to pass into `mod.predict`.

    Returns
    -------
    np.ndarray
        Probabilities in the range [0, 1] obtained by applying the logistic transformation.

    Raises
    ------
    AttributeError
        If `mod` does not implement a `predict` method.
    ValueError
        If the output of `mod.predict(x)` cannot be converted to a NumPy array.
    """
    if not hasattr(mod, "predict"):
        raise AttributeError("Provided model object has no 'predict' method.")

    pred = mod.predict(x)

    try:
        pred = np.asarray(pred, dtype=float)
    except Exception as e:
        raise ValueError(f"Could not convert model predictions to array: {e}")

    prob = 1 / (1 + np.exp(-pred))
    return prob

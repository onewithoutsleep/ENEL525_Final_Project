"""The three models. Read this file first to see how each model is created."""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import RANDOM_SEED, VOLATILITY_FLOOR


def build_decision_tree(settings):
    """One tree learns rules such as 'is recent volatility above this value?'"""
    return DecisionTreeRegressor(random_state=RANDOM_SEED, **settings)


def build_random_forest(settings):
    """Many different trees make predictions; their predictions are averaged."""
    return RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,  # Use available CPU cores. Change to 1 to use fewer resources.
        **settings,
    )


def log_volatility(values):
    """Handle genuine zero-volatility days without taking log(0)."""
    return np.log(np.maximum(values, VOLATILITY_FLOOR))


def build_neural_network(settings):
    """A feedforward multilayer perceptron (MLP) for a numeric prediction."""
    network = MLPRegressor(
        activation="relu",
        solver="adam",
        random_state=RANDOM_SEED,
        shuffle=False,
        # Built-in early stopping would create a random validation split.
        # We instead evaluate on the later date window in train.py.
        early_stopping=False,
        **settings,
    )

    # Volume and returns have very different scales. Learn the scaling only
    # from training rows, and reuse it unchanged for validation and prediction.
    scaled_network = Pipeline([
        ("scale_inputs", StandardScaler()),
        ("network", network),
    ])

    # Learn log(volatility), then exponentiate predictions back to volatility.
    # This gives positive predictions and avoids fitting tiny decimal targets.
    # It minimizes squared LOG-volatility error, not QLIKE directly.
    return TransformedTargetRegressor(
        regressor=scaled_network,
        func=log_volatility,
        inverse_func=np.exp,
        check_inverse=False,  # The numerical floor slightly changes zero values.
    )

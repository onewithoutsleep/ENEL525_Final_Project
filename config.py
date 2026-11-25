"""Settings you can change before training. Start with the defaults."""

RANDOM_SEED = 42  # Makes repeated runs easier to compare.
LOOKBACK_DAYS = 20  # Use today and the previous 19 trading days as inputs.
BASELINE_DAYS = 20  # The proposal's moving-average comparison.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20  # The remaining data is reserved for testing.
MINIMUM_EXAMPLES = 100  # A starter-code guard, NOT a guarantee of enough data.

# Optional wider-market inputs. Use train.py --market-data <price folder>.
# The target symbol is omitted automatically (SPY should not get SPY twice).
MARKET_SYMBOLS = [
    "GLD", "SLV", "USO", "UNG", "CPER", "DBA", "URA",
    "SPY", "QQQ", "DIA", "IWM", "VTI", "EFA", "EEM", "TLT", "XLF", "XLE",
]
MARKET_LOOKBACK_DAYS = 20
# Use the PREVIOUS completed session's daily bar: daily bars can be revised
# after the stock-market close. Never use today's finalized daily bar at 4 pm.
MARKET_DELAY_SESSIONS = 1

# Each dictionary describes one possible model. Normally only the first is used.
# Run train.py with --tune to try every dictionary and choose on validation data.
TREE_SETTINGS = [
    {"max_depth": 4, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 6, "min_samples_leaf": 5, "min_samples_split": 10},
]

FOREST_SETTINGS = [
    {"n_estimators": 150, "max_depth": 6,
     "min_samples_leaf": 5, "min_samples_split": 10},
    {"n_estimators": 250, "max_depth": 10,
     "min_samples_leaf": 10, "min_samples_split": 20},
]

NETWORK_SETTINGS = [
    {"hidden_layer_sizes": (32, 16), "learning_rate_init": 0.001,
     "batch_size": 32, "alpha": 0.01, "max_iter": 1000},
    {"hidden_layer_sizes": (16,), "learning_rate_init": 0.0005,
     "batch_size": 16, "alpha": 0.01, "max_iter": 1000},
]

# Hidden layers (32, 16) means 32 neurons followed by 16 neurons.
# The number of INPUT neurons is inferred from your feature columns at fit time.
# max_iter limits training passes, not the number of training examples.

VOLATILITY_FLOOR = 1e-8  # Avoid log(0) when training the neural network.
VARIANCE_FLOOR = 1e-12  # Avoid division by zero / log(0) in QLIKE.

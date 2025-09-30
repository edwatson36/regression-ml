import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Below code to be turned into a function grid_srch_rf()
# Consider using from sklearn.model_selection import GridSearchCV rather than the for loop for better efficiency
# Consider how to input a configurable set of regularisation parameters (not just fixed with 'min_samples_leaf' and 'max_depth'

# Define parameters and values to grid search on
param_grid = {
    'min_samples_leaf': [None, 5, 10],  # None equivalent to 1 (default)
    'max_depth': [None, 5, 10, 20]         # None for no constraint
}

# Variables to store the best model and highest OOB score
best_oob_score = -np.inf  # Initialize to a very low value
best_params = None
best_rf = None

# Perform Grid Search
for min_samples_leaf in param_grid['min_samples_leaf']:
    for max_depth in param_grid['max_depth']:
        # Define and fit the model with current parameters
        rf = RandomForestRegressor(
            oob_score=True,
            min_samples_leaf=min_samples_leaf if min_samples_leaf else 1,
            max_depth=max_depth,
            random_state=42
        )

        # Fit the model on training data
        rf.fit(x_train_bf, y_train)

        # Calculate OOB score
        oob_score = rf.oob_score_

        # Update best model if this is the highest OOB score
        if oob_score > best_oob_score:
            best_oob_score = oob_score
            best_params = {'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
            best_rf = rf

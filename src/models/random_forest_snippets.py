"""
Example: Best parameters search using GridSearchCV
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from evaluation.regression_utils import reg_to_prob # custom function converting regression output to (0,1) classification.

# Example parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor() # or RandomForestClassifier
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)  # X_train, y_train should be defined elsewhere

# Best estimator
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# Obtain prediction from train set
pred_rf = reg_to_prob(best_rf, x_train) # if RandomForestRegressor but want classification
pred_rf = best_rf.predict_proba(x_train) # if RandomForestClassifier

"""
Example: Custom grid search using oob_score
Using GridSearchCV is better than this
"""

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

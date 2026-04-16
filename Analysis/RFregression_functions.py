import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def fit_RFregression(
    X,
    y,
    cat_cols=None,
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    stopping_rounds=50,
    n_repeats=20,
    plot_pred=True,
    plot_importance=True):
    """
    Fit a LightGBM regressor and return model outputs + diagnostics.

    Returns a dictionary containing:
    - fitted model
    - predictions
    - RMSE and R2
    - split importance
    - gain importance
    - permutation importance
    - combined importance table
    """

    if cat_cols is None:
        cat_cols = []

    # Fit model
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )

    model.fit(
        X,
        y,
        categorical_feature=cat_cols,
        eval_set=[(X, y)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    )

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("RMSE:", rmse)
    print("R2:", r2)

    # Split importance
    split_importance = pd.Series(
        model.feature_importances_,
        index=X.columns,
        name="split_importance"
    ).sort_values(ascending=False)

    split_importance_percent = (
        split_importance / split_importance.sum()
    ).rename("split_importance_percent")

    # Gain importance
    gain_importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
        name="gain_importance"
    ).sort_values(ascending=False)

    gain_importance_percent = (
        gain_importance / gain_importance.sum()
    ).rename("gain_importance_percent")

    # Permutation importance
    perm_result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    permutation_importance_mean = pd.Series(
        perm_result.importances_mean,
        index=X.columns,
        name="permutation_importance_mean"
    ).sort_values(ascending=False)

    permutation_importance_std = pd.Series(
        perm_result.importances_std,
        index=X.columns,
        name="permutation_importance_std"
    )

    # Combined table
    importance_table = pd.concat(
        [
            split_importance,
            split_importance_percent,
            gain_importance,
            gain_importance_percent,
            permutation_importance_mean,
            permutation_importance_std
        ],
        axis=1
    )

    # Observed vs predicted plot
    if plot_pred:
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))

        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_pred, s=5, alpha=0.5, c="black")
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", c="red")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title("Observed vs Predicted")
        plt.tight_layout()
        plt.show()

    # Importance plots
    if plot_importance:
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))

        # Split importance
        split_importance.plot.bar(ax=axes[0])
        axes[0].set_title("Split Importance")
        axes[0].set_ylabel("Split Count")
        axes[0].tick_params(axis="x", rotation=90)

        # Gain importance
        gain_importance.plot.bar(ax=axes[1])
        axes[1].set_title("Gain Importance")
        axes[1].set_ylabel("Total Gain")
        axes[1].tick_params(axis="x", rotation=90)

        # Permutation importance
        permutation_importance_mean.plot.bar(ax=axes[2])
        axes[2].set_title("Permutation Importance")
        axes[2].set_ylabel("Mean Importance")
        axes[2].tick_params(axis="x", rotation=90)

        plt.tight_layout()
        plt.show()

    results = {
        "model": model,
        "predictions": y_pred,
        "rmse": rmse,
        "r2": r2,
        "split_importance": split_importance,
        "split_importance_percent": split_importance_percent,
        "gain_importance": gain_importance,
        "gain_importance_percent": gain_importance_percent,
        "permutation_importance_mean": permutation_importance_mean,
        "permutation_importance_std": permutation_importance_std,
        "importance_table": importance_table
    }

    return results

def plot_importance(results):

    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    results["split_importance"].sort_values(ascending=False).plot.bar(ax=axes[0])
    axes[0].set_title("Split Importance")

    results["gain_importance"].sort_values(ascending=False).plot.bar(ax=axes[1])
    axes[1].set_title("Gain Importance")

    results["permutation_importance_mean"].sort_values(ascending=False).plot.bar(ax=axes[2])
    axes[2].set_title("Permutation Importance")

    for ax in axes:
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()
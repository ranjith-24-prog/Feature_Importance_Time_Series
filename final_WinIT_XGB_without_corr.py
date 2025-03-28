import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


# Train the XGBoost model
def train_xgboost(X_train, y_train, num_rounds=100):
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=num_rounds)
    model.fit(X_train, y_train)
    return model


# Compute WinIT feature importance
def compute_winit(model, X_train, y_train, X_test, y_test, feature_names):
    baseline_loss = np.mean((model.predict(X_test) - y_test) ** 2)
    print(f"Baseline Loss: {baseline_loss}")
    feature_importance = {}

    for feature_idx, feature_name in enumerate(feature_names):
        print(f"Processing feature: {feature_name}")
        
        # Replace feature with its mean to simulate removal
        feature_mean = np.mean(X_test[:, feature_idx])
        X_test_modified = X_test.copy()
        X_test_modified[:, feature_idx] = feature_mean

        modified_loss = np.mean((model.predict(X_test_modified) - y_test) ** 2)

        # Calculate importance as the increase in loss
        importance = modified_loss - baseline_loss
        feature_importance[feature_name] = max(importance, 0)  # Ensure non-negative importance

        print(f"Feature: {feature_name}, Modified Loss: {modified_loss}, Importance: {importance}")

    # Normalize importance
    max_importance = max(feature_importance.values(), default=1)  # Avoid division by zero
    feature_importance = {k: v / max_importance for k, v in feature_importance.items()}

    return feature_importance


def process_dataset_with_winit_xgb_without_corr(path, prepare_data_func):
    # Prepare data
    X, y, retained_features = prepare_data_func(path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Compute WinIT feature importance
    feature_importance = compute_winit(model, X_train, y_train, X_test, y_test, retained_features)

    # Save results
    dataset_name = os.path.basename(path).split(".")[0]
    importance_df = pd.DataFrame({
        "Feature": feature_importance.keys(),
        "Importance": feature_importance.values()
    }).sort_values(by="Importance", ascending=False)
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison

    csv_output_path = f"output/FI_Dataframes/WinIT/{dataset_name}_WinIT_XGB_without_corr.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    importance_df.to_csv(csv_output_path, index=False)

    # Plot feature importance
    plot_path = f"output/FI_Plots/WinIT/{dataset_name}_WinIT_XGB_without_corr.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel("Normalized Importance")
    plt.ylabel("Feature")
    plt.title(f"WinIT Feature Importance (Sorted by Importance) - {dataset_name}")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()
    print(f"Processed and saved results for {dataset_name}")

    return {
        "test_loss": rmse, 
        "top_features": top_features  
    }
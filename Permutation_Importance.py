import numpy as np
import os
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from model import LSTM_MODEL
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

import prepare_data

def Permutation_Importance_FI():
    data, targets = prepare_data.prepare_data()
    # Reshape features for time series (samples, time_steps, features)
    # For example, use a sliding window to create time steps
    time_steps = 5  # Define the number of time steps
    X = np.array([data[i:i + time_steps] for i in range(len(data) - time_steps)])
    y = targets[time_steps:]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # ---------- 2. Define LSTM Model ----------
    model = LSTM_MODEL(X_train,X_test,y_train,y_test)

    # ---------- 4. Feature Importance Techniques ----------

    # Permutation Importance (requires model performance degradation calculation)
    def permutation_importance_lstm(model, X, y, loss_fn, feature_idx):
        """Calculates permutation importance for a single feature."""
        X_perm = X.clone()
        X_perm[:, :, feature_idx] = X_perm[torch.randperm(X_perm.size(0)), :, feature_idx]
        with torch.no_grad():
            y_pred = model(X_perm)
            loss_perm = loss_fn(y_pred, y)
        return loss_perm.item()

    feature_names = prepare_data.get_used_features()
    # Compute importance for each feature
    importances = []
    criterion = nn.MSELoss()
    input_size = 52
    for i in range(input_size):
        baseline_loss = criterion(model(X_test), y_test).item()
        perm_loss = permutation_importance_lstm(model, X_test, y_test, criterion, i)
        importances.append(perm_loss - baseline_loss)

    # Combine feature names with importance values
    feature_importance_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Permutation Importance": importances
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by="Permutation Importance", ascending=False)

    # Print the feature importance
    print(feature_importance_df)

    # Plot with adjusted Y-axis spacing and name rotation
    plt.figure(figsize=(10, 12))  # Adjust figure size for better visibility
    plt.barh(feature_importance_df["Feature Name"], feature_importance_df["Permutation Importance"])
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature Name")
    plt.title("LSTM Feature Importance (Permutation)")
    plt.tight_layout()
    plt.savefig("output/Permutation_importance_plot.png", format="png", dpi=300)
    plt.show()

    output_path = os.path.join("output/FI_Dataframes", "Permutation_Importance.csv")
    feature_importance_df.to_csv(output_path, index=False)
    return feature_importance_df

import numpy as np
import os
import pandas as pd
import torch
from model import LSTM_MODEL
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

import prepare_data

def Integrated_Gradients_FI():
    # ---------- 1. Data Preparation ----------
    features_data, targets_data = prepare_data.prepare_data()
    # Reshape features for time series (samples, time_steps, features)
    time_steps = 5  # Define the number of time steps
    X = np.array([features_data[i:i + time_steps] for i in range(len(features_data) - time_steps)])
    y = targets_data[time_steps:]

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # ---------- 2. Define LSTM Model ----------
    model = LSTM_MODEL(X_train,X_test,y_train,y_test)

    # ---------- 4. Integrated Gradients with Batching ----------
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model)

    # Function to compute Integrated Gradients in batches
    def compute_ig_in_batches(ig, inputs, baselines, batch_size, target):
        all_attributions = []
        for i in range(0, inputs.size(0), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_baselines = baselines[i : i + batch_size]
            attributions, _ = ig.attribute(batch_inputs, baselines=batch_baselines, target=target, return_convergence_delta=True)
            all_attributions.append(attributions)
        return torch.cat(all_attributions)

    # Define baseline (zero vector with same shape as input)
    baseline = torch.zeros_like(X_test)

    # Batch size to control memory usage
    batch_size = 32  # Adjust as needed to fit your memory constraints

    # Compute Integrated Gradients attributions in smaller batches
    attributions = compute_ig_in_batches(ig, X_test, baseline, batch_size=batch_size, target=0)

    # Aggregate feature importance across time steps (absolute mean attribution)
    feature_importance = attributions.abs().mean(dim=0).mean(dim=0).detach().numpy()

    # ---------- 5. Visualize Feature Importance ----------
    # Feature names (replace with actual feature names if available)
    feature_names = prepare_data.get_used_features()

    # Combine feature names with importance values
    feature_importance_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Integrated Gradients Importance": feature_importance
    })

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by="Integrated Gradients Importance", ascending=False)

    # Print the feature importance
    print(feature_importance_df)

    # Plot the Feature Importance
    plt.figure(figsize=(10, 12))  # Adjust figure size for better visibility
    plt.barh(feature_importance_df["Feature Name"], feature_importance_df["Integrated Gradients Importance"])
    plt.xlabel("Integrated Gradients Importance")
    plt.ylabel("Feature Name")
    plt.title("LSTM Feature Importance (Integrated Gradients)")
    plt.tight_layout()  # Automatically adjust layout to avoid overlap
    plt.gca().invert_yaxis()  # Invert Y-axis for better readability
    plt.savefig("output/Integrated_Gradients_plot.png", format="png", dpi=300)
    plt.show()

    output_path = os.path.join("output/FI_Dataframes", "Integrated_Gradients.csv")
    feature_importance_df.to_csv(output_path, index=False)
    return feature_importance_df
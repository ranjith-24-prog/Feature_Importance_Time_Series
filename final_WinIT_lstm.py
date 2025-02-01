import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import models_final


# Train the baseline model
def train_model(model, X_train, y_train, optimizer, criterion, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    return model

# Compute WinIT feature importance
def compute_winit(model, X_train, y_train, X_test, y_test, criterion, feature_names):
    baseline_loss = criterion(model(X_test), y_test).item()
    print(f"Baseline Loss: {baseline_loss}")
    feature_importance = {}

    for feature_idx, feature_name in enumerate(feature_names):
        print(f"Processing feature: {feature_name}")
        
        # Replace feature with its mean to simulate removal
        feature_mean = X_test[:, :, feature_idx].mean().item()
        X_test_modified = X_test.clone()
        X_test_modified[:, :, feature_idx] = feature_mean

        with torch.no_grad():
            modified_loss = criterion(model(X_test_modified), y_test).item()
        
        # Calculate importance as the increase in loss
        importance = modified_loss - baseline_loss
        feature_importance[feature_name] = max(importance, 0)  # Ensure non-negative importance

        print(f"Feature: {feature_name}, Modified Loss: {modified_loss}, Importance: {importance}")

    # Normalize importance
    max_importance = max(feature_importance.values(), default=1)  # Avoid division by zero
    feature_importance = {k: v / max_importance for k, v in feature_importance.items()}

    return feature_importance

# Process dataset with WinIT
def process_dataset_with_winit_lstm_with_corr(path, prepare_data_func):
    # Prepare data
    X, y, retained_features = prepare_data_func(path)
    input_size = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Define and train the model
    hidden_size = 128
    output_size = y_train.size(-1)
    num_layers = 2
    model = models_final.LSTMModel(input_size, hidden_size, output_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model = train_model(model, X_train, y_train, optimizer, criterion)

    # Compute WinIT feature importance
    feature_importance = compute_winit(model, X_train, y_train, X_test, y_test, criterion, retained_features)

    # Save results
    dataset_name = os.path.basename(path).split(".")[0]
    importance_df = pd.DataFrame({
        "Feature": feature_importance.keys(),
        "Importance": feature_importance.values()
    }).sort_values(by="Importance", ascending=False)
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison

    csv_output_path = f"output/FI_Dataframes/WinIT/{dataset_name}_WinIT_WithCorr_without_redundancy.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    importance_df.to_csv(csv_output_path, index=False)

    # Plot feature importance
    plot_path = f"output/FI_Plots/WinIT/{dataset_name}_WinIT_WithCorr_without_redundancy.png"
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
        "test_loss": criterion(model(X_test), y_test).item(), 
        "top_features": top_features  
    }

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import models_final
import os


# Train VAE
def train_vae(X_train, input_size, hidden_size=64, latent_size=16, num_epochs=50, batch_size=64, lr=0.001):
    vae = models_final.VAE(input_size, hidden_size, latent_size)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train.squeeze(1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch in dataloader:
            batch_data = batch[0]
            reconstructed, mu, logvar = vae(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    return vae


# Compute WinIT with VAE
def compute_winit_with_vae(model, vae, X_test, y_test, criterion, feature_names):
    baseline_loss = criterion(model(X_test), y_test).item()
    print(f"Baseline Loss: {baseline_loss}")
    feature_importance = {}

    for feature_idx, feature_name in enumerate(feature_names):
        print(f"Processing feature: {feature_name}")

        # Use VAE to generate replacements for the feature
        X_test_modified = X_test.clone()
        X_test_flat = X_test_modified.squeeze(1)
        with torch.no_grad():
            reconstructed = vae(X_test_flat)[0]
        # Fix dimension mismatch here
        X_test_modified[:, :, feature_idx] = reconstructed[:, feature_idx].unsqueeze(1)

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


def process_dataset_with_winit_lstm_vae_without_corr(path, prepare_data_func):
    X_scaled, y_scaled, feature_names = prepare_data_func(path)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add time dimension
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Train VAE for feature replacement
    input_size = X_train.size(-1)
    vae = train_vae(X_train, input_size)

    # Train LSTM model
    hidden_size = 128
    output_size = y_train.size(-1)
    num_layers = 2
    model = models_final.LSTMModel(input_size, hidden_size, output_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Compute WinIT with VAE
    feature_importance = compute_winit_with_vae(model, vae, X_test, y_test, criterion, feature_names)

    # Save results
    dataset_name = os.path.basename(path).split(".")[0]
    importance_df = pd.DataFrame({
        "Feature": feature_importance.keys(),
        "Importance": feature_importance.values()
    }).sort_values(by="Importance", ascending=False)
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison

    csv_output_path = f"output/FI_Dataframes/WinIT/{dataset_name}_WinIT_lstm_Without_Corr_vae.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    importance_df.to_csv(csv_output_path, index=False)

    # Visualize feature importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    sorted_feature_names = [f[0] for f in sorted_features]
    sorted_importance_values = [f[1] for f in sorted_features]

    plot_path = f"output/FI_Plots/WinIT/{dataset_name}_WinIT_lstm_Without_Corr_vae.png"
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_feature_names, sorted_importance_values, color='skyblue')
    plt.xlabel("Features")
    plt.ylabel("Normalized Importance")
    plt.title(f"WinIT Feature Importance (Using VAE for Feature Replacement) - {dataset_name}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()
    print(f"Processed and saved results for {dataset_name}")

    return {
        "test_loss": criterion(model(X_test), y_test).item(), 
        "top_features": top_features  
    }
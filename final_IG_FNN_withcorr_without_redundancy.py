import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import models_final
from prepare_data import get_all_features
import os

# Helper function to compute IG in batches
def compute_ig_in_batches(ig, inputs, baselines, batch_size, target_idx):
    all_attributions = []
    for i in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_baselines = baselines[i:i + batch_size]
        attributions = ig.attribute(batch_inputs, baselines=batch_baselines, target=target_idx)
        all_attributions.append(attributions)
    return torch.cat(all_attributions)


def process_dataset_with_IG_FNN_with_corr_without_redundancy(path, prepare_data_func):
    X, y, retained_features = prepare_data_func(path)
    all_features = get_all_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Initialize FNN model
    input_size = X_train.size(-1)
    hidden_size = 128  # Number of neurons in each hidden layer
    output_size = y_train.size(-1)  # Number of output variables
    num_layers = 3  # Number of hidden layers
    model = models_final.FNNModel(input_size, hidden_size, output_size, num_layers)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the FNN model
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # Evaluate the FNN model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # Apply Integrated Gradients
    ig = IntegratedGradients(model)

    # Compute attributions using Integrated Gradients
    baseline = X_test.mean(dim=0, keepdim=True).expand_as(X_test)
    batch_size = 32
    attributions = []
    for target_idx in range(y_train.size(1)):
        target_attributions = compute_ig_in_batches(ig, X_test, baseline, batch_size, target_idx)
        attributions.append(target_attributions.abs().mean(dim=0))
    collective_attributions = torch.stack(attributions).mean(dim=0).detach().numpy()

    # Map feature importance back to all features
    feature_importance = {feature: 0 for feature in all_features}  # Initialize all features with 0 importance
    for feature, importance in zip(retained_features, collective_attributions):
        feature_importance[feature] = importance

    # Save results
    dataset_name = os.path.basename(path).split(".")[0]
    importance_df = pd.DataFrame({
        "Feature": feature_importance.keys(),
        "Importance": feature_importance.values()
    }).sort_values(by="Importance", ascending=False)
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison

    csv_output_path = f"output/FI_Dataframes/Integrated_Gradients/{dataset_name}_IG_FNN_withcorr_without_redundancy.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    importance_df.to_csv(csv_output_path, index=False)

    # Plot feature importance
    plot_path = f"output/FI_Plots/Integrated_Gradients/{dataset_name}_IG_FNN_withcorr_without_redundancy.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.figure(figsize=(12, max(6, len(importance_df) * 0.5)))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance (Integrated Gradients) - {dataset_name}")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig(plot_path, format="png", dpi=300)
    plt.close()
    print(f"Processed and saved results for {dataset_name}")

    return {
        "test_loss": criterion(model(X_test), y_test).item(), 
        "top_features": top_features  
    }
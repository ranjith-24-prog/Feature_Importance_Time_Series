import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from prepare_data_new import prepare_data_with_correlation,get_all_features
import models_final
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


# Prepare data
X, y, retained_features = prepare_data_with_correlation()
all_features = get_all_features()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Initialize LSTM model
input_size = X_train.size(-1)
hidden_size = 128
output_size = y_train.size(-1)
num_layers = 2
model = models_final.LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train model
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Apply Integrated Gradients
ig = IntegratedGradients(model)

baseline = X_test.mean(dim=0, keepdim=True).expand_as(X_test)
batch_size = 32
attributions = []
for target_idx in range(y_train.size(1)):
    target_attributions = compute_ig_in_batches(ig, X_test, baseline, batch_size, target_idx)
    attributions.append(target_attributions.abs().mean(dim=0))
collective_attributions = torch.stack(attributions).mean(dim=0).mean(dim=0).detach().numpy()

# Map feature importance back to all features
feature_importance = {feature: 0 for feature in all_features}  # Initialize all features with 0 importance
for feature, importance in zip(retained_features, collective_attributions):
    feature_importance[feature] = importance

# Create DataFrame and improved plot
importance_df = pd.DataFrame({
    "Feature": feature_importance.keys(),
    "Importance": feature_importance.values()
}).sort_values(by="Importance", ascending=False)
csv_output_path = f"output/FI_Dataframes/Integrated_Gradients/IG_LSTM_WithCorr.csv"
importance_df.to_csv(csv_output_path, index=False)
print(f"Feature importance saved successfully to: {csv_output_path}")

# Improved plot for better readability
plot_path = f"output/FI_Plots/Integrated_Gradients/IG_LSTM_WithCorr.png"
print(f"Saving plot to: {plot_path}")
plt.figure(figsize=(12, max(6, len(importance_df) * 0.5)))  # Dynamically adjust height
plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance (Integrated Gradients)", fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.savefig(plot_path, format="png", dpi=300)
print(f"Plot saved successfully: {os.path.exists(plot_path)}")
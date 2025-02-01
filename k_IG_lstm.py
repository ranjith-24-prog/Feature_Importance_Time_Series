import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# Helper function to compute IG in batches
def compute_ig_in_batches(ig, inputs, baselines, batch_size, target_idx):
    all_attributions = []
    for i in range(0, inputs.size(0), batch_size):
        batch_inputs = inputs[i:i + batch_size]
        batch_baselines = baselines[i:i + batch_size]
        attributions = ig.attribute(batch_inputs, baselines=batch_baselines, target=target_idx)
        all_attributions.append(attributions)
    return torch.cat(all_attributions)

# Load and preprocess the dataset
def prepare_data():
    path = "static/Dataset/DMC2_AL_CP2.csv"
    df = pd.read_csv(path)

    # Feature and target selection
    all_features = [
        'LOAD|1', 'LOAD|2', 'LOAD|3', 'LOAD|6',
        'ENC_POS|1', 'ENC_POS|2', 'ENC_POS|3', 'ENC_POS|6',
        'CTRL_DIFF2|1', 'CTRL_DIFF2|2', 'CTRL_DIFF2|3', 'CTRL_DIFF2|6',
        'TORQUE|1', 'TORQUE|2', 'TORQUE|3', 'TORQUE|6',
        'DES_POS|1', 'DES_POS|2', 'DES_POS|3', 'DES_POS|6',
        'CTRL_DIFF|1', 'CTRL_DIFF|2', 'CTRL_DIFF|3', 'CTRL_DIFF|6',
        'CTRL_POS|1', 'CTRL_POS|2', 'CTRL_POS|3', 'CTRL_POS|6',
        'VEL_FFW|1', 'VEL_FFW|2', 'VEL_FFW|3', 'VEL_FFW|6',
        'CONT_DEV|1', 'CONT_DEV|2', 'CONT_DEV|3', 'CONT_DEV|6',
        'CMD_SPEED|1', 'CMD_SPEED|2', 'CMD_SPEED|3', 'CMD_SPEED|6',
        'TORQUE_FFW|1', 'TORQUE_FFW|2', 'TORQUE_FFW|3', 'TORQUE_FFW|6',
        'ENC1_POS|1', 'ENC1_POS|2', 'ENC1_POS|3', 'ENC1_POS|6'
    ]
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Compute correlation matrix
    correlation_matrix = X.corr()

    # Find highly correlated feature pairs
    high_corr_features = [
        (col1, col2, correlation_matrix.loc[col1, col2])
        for col1 in correlation_matrix.columns
        for col2 in correlation_matrix.columns
        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
    ]

    # Print correlation values and feature pairs
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    print("\nHighly Correlated Features (correlation > 0.9):")
    for col1, col2, corr_value in high_corr_features:
        print(f"{col1} and {col2} with correlation {corr_value:.2f}")

    # Drop one feature from each highly correlated pair
    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    print('fetures to drop:', features_to_drop)
    retained_features = [f for f in all_features if f not in features_to_drop]
    X = X[retained_features]

    return X, y, all_features, retained_features

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Fully connected layer at the last time step
        return out

# Prepare data
X, y, all_features, retained_features = prepare_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Initialize LSTM model
input_size = X_train.size(-1)
hidden_size = 128
output_size = y_train.size(-1)
num_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
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

print("\nFeature Importance (Integrated Gradients):")
print(importance_df)

# Improved plot for better readability
plt.figure(figsize=(12, max(6, len(importance_df) * 0.5)))  # Dynamically adjust height
plt.barh(importance_df["Feature"], importance_df["Importance"], color='skyblue')
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Feature Importance (Integrated Gradients)", fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

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

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, all_features

# Prepare data
X, y, feature_names = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

# Initialize LSTM model
input_size = X_train.shape[1]
hidden_size = 128
output_size = y_train.shape[1]
num_layers = 2
model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train model
print("Training LSTM model...")
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1))
    loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1))
    test_loss = criterion(test_outputs, torch.tensor(y_test, dtype=torch.float32))
    print(f"Test Loss: {test_loss.item():.4f}")

# Apply LIME
def predict_fn(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
    return model(input_tensor).detach().numpy()

explainer = LimeTabularExplainer(X_train, mode='regression', feature_names=feature_names)

sample_idx = 0
exp = explainer.explain_instance(X_test[sample_idx], predict_fn)

# Extract feature importance and plot
feature_importance_dict = dict(exp.as_map()[1])  # Get feature indices and importance scores
feature_importance = {feature_names[i]: abs(importance) for i, importance in feature_importance_dict.items()}

print("\nFeature Importance:")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(12, 6))
plt.barh(list(feature_importance.keys()), list(feature_importance.values()), color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (LIME)')
plt.gca().invert_yaxis()
plt.show()

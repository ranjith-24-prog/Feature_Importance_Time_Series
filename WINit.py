import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import prepare_data


def WINit_FI():

    # ---------- 1. Data Preparation ----------
    features_data, targets_data = prepare_data.prepare_data()
    # Reshape features for time series (samples, time_steps, features)
    time_steps = 5  # Define the number of time steps
    X = np.array([features_data[i:i + time_steps] for i in range(len(features_data) - time_steps)])
    y = targets_data[time_steps:]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors for LSTM
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)


    # ---------- 2. Define LSTM Model ----------
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)  # Pass through LSTM
            out = self.fc(out[:, -1, :])  # Fully connected on the last time step
            return out


    # Model Parameters
    input_size = features_data.shape[1]  # Number of features
    hidden_size = 64  # LSTM hidden units
    output_size = y_train.shape[1]  # Number of targets
    num_layers = 2  # Number of LSTM layers

    # Initialize the model, loss function, and optimizer
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # ---------- 3. Train the Model ----------
    def train_model(model, X_train, y_train, X_test, y_test, num_epochs=20):
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
        return test_loss


    # Train the baseline model
    baseline_loss = train_model(model, X_train, y_train, X_test, y_test)


    # ---------- 4. WinIT Feature Importance ----------
    # Function to compute feature importance with iterative feature removal
    def compute_winit_importance(model, X_train, y_train, X_test, y_test, used_features):
        feature_importance = {}
        for feature_idx, feature_name in enumerate(used_features):
            # Remove one feature (set to zero for all time steps)
            X_train_modified = X_train.clone()
            X_test_modified = X_test.clone()
            X_train_modified[:, :, feature_idx] = 0
            X_test_modified[:, :, feature_idx] = 0

            # Retrain the model and compute test loss
            modified_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
            modified_model.load_state_dict(model.state_dict())  # Start with same weights as baseline
            modified_optimizer = torch.optim.Adam(modified_model.parameters(), lr=0.001)

            modified_loss = train_model(modified_model, X_train_modified, y_train, X_test_modified, y_test)

            # Calculate feature importance as the increase in loss
            feature_importance[feature_name] = modified_loss - baseline_loss

        return feature_importance


    # Compute feature importance using WinIT
    used_features = prepare_data.get_used_features()
    feature_importance = compute_winit_importance(model, X_train, y_train, X_test, y_test, used_features)

    # ---------- 5. Visualize Feature Importance ----------
    # Convert feature importance to DataFrame
    feature_importance_df = pd.DataFrame({
        "Feature Name": list(feature_importance.keys()),
        "WinIT Importance": list(feature_importance.values())
    }).sort_values(by="WinIT Importance", ascending=False)

    # Improved Visualization
    plt.style.use("ggplot")
    plt.figure(figsize=(14, 16))
    bars = plt.barh(
        feature_importance_df["Feature Name"],
        feature_importance_df["WinIT Importance"],
        color="lightcoral", edgecolor="black", linewidth=0.7
    )

    # Add data labels to each bar
    for bar in bars:
        plt.text(
            bar.get_width() + 0.0001,  # Position slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Center vertically
            f"{bar.get_width():.4f}",  # Format the value
            va="center", fontsize=10
        )

    # Adjust plot aesthetics
    plt.xlabel("WinIT Importance (Change in Loss)", fontsize=14, weight="bold")
    plt.ylabel("Feature Name", fontsize=14, weight="bold")
    plt.title("LSTM Feature Importance (WinIT)", fontsize=16, weight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Show the plot
    plt.gca().invert_yaxis()
    plt.savefig("output/WINit_plot.png", format="png", dpi=300)
    plt.show()
    output_path = os.path.join("output/FI_Dataframes", "WINit.csv")
    feature_importance_df.to_csv(output_path, index=False)
    return feature_importance_df

WINit_FI()
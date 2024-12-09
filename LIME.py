import numpy as np
import pandas as pd
import torch
from model import LSTM_MODEL_LIME
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

import prepare_data

def LIME_FI():
    # ---------- 1. Data Preparation ----------
    features_data, targets_data = prepare_data.prepare_data()

    # Reshape features for time series (samples, time_steps, features)
    time_steps = 5  # Define the number of time steps
    X = np.array([features_data[i:i + time_steps] for i in range(len(features_data) - time_steps)])
    y = targets_data[time_steps:]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Flatten X for LIME (since LIME works with 2D tabular data)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Convert data to PyTorch tensors for LSTM
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # ---------- 2. Define LSTM Model ----------
    input_size = features_data.shape[1]
    model = LSTM_MODEL_LIME(X_train,X_test,y_train,y_test,features_data)

    # ---------- 4. LIME Explanations ----------
    # Generate feature names for flattened data (expand across time steps)
    used_features = prepare_data.get_used_features()
    feature_names = [f"{feature}_t{t}" for t in range(time_steps) for feature in used_features]

    # Define a wrapper to predict using the trained PyTorch model
    def predict_function(inputs):
        inputs = torch.tensor(inputs.reshape(-1, time_steps, input_size), dtype=torch.float32)
        with torch.no_grad():
            outputs = model(inputs).numpy()
        return outputs

    # Initialize LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train_flat,  # Use flattened training data for LIME
        mode='regression',  # Specify regression mode
        feature_names=feature_names,  # Correctly flattened feature names
        verbose=True
    )

    # Compute feature importance for a sample instance
    instance_idx = 0  # Change this to test different instances
    explanation = lime_explainer.explain_instance(
        data_row=X_test_flat[instance_idx],
        predict_fn=predict_function,
        num_features=10  # Number of top features to display
    )



    # ---------- 5. Aggregate Feature Importance Across Dataset ----------
    # Initialize feature importance array
    feature_importance = np.zeros(X_train_flat.shape[1])

    # Loop over all test instances and accumulate importance scores
    for i in range(X_test_flat.shape[0]):
        explanation = lime_explainer.explain_instance(
            data_row=X_test_flat[i],
            predict_fn=predict_function,
            num_features=10
        )
        for feature, importance in explanation.as_map()[1]:
            feature_importance[feature] += abs(importance)

    # Normalize importance values
    feature_importance /= X_test_flat.shape[0]

    # Combine flattened feature names with importance values
    feature_importance_df = pd.DataFrame({
        "Feature Name": feature_names,
        "LIME Importance": feature_importance
    }).sort_values(by="LIME Importance", ascending=False)

    # ---------- 6. Aggregate Feature Importance Across Time Steps ----------
    # Combine time-step feature importance into original features

    aggregated_importance = np.zeros(len(used_features))

    for i, feature in enumerate(used_features):
        # Sum importance across time steps
        aggregated_importance[i] = sum(
            feature_importance[j] for j in range(i, len(feature_importance), len(used_features))
        )

    # Create a DataFrame for aggregated importance
    aggregated_feature_importance_df = pd.DataFrame({
        "Feature Name": used_features,
        "Aggregated LIME Importance": aggregated_importance
    }).sort_values(by="Aggregated LIME Importance", ascending=False)

    # ---------- 7. Visualize Feature Importance ----------
    # Plot the aggregated feature importance
    plt.figure(figsize=(10, 12))
    plt.barh(aggregated_feature_importance_df["Feature Name"], aggregated_feature_importance_df["Aggregated LIME Importance"])
    plt.xlabel("Aggregated LIME Importance")
    plt.ylabel("Feature Name")
    plt.title("LSTM Feature Importance (Aggregated LIME)")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.savefig("output/LIME_plot.png", format="png", dpi=300)
    plt.show()
    output_path = os.path.join("output/FI_Dataframes", "LIME.csv")
    aggregated_feature_importance_df.to_csv(output_path, index=False)
    return aggregated_feature_importance_df
LIME_FI()
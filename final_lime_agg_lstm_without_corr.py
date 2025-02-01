import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from collections import defaultdict
import models_final
import os

def process_dataset_with_lime_agg_lstm_without_corr(path, prepare_data_func):
    def predict_fn(input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
        return model(input_tensor).detach().numpy()

    X, y, feature_names = prepare_data_func(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models_final.LSTMModel(input_size=X_train.shape[1], hidden_size=128, output_size=y_train.shape[1], num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    print("Training LSTM model...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1))
        loss = criterion(outputs, torch.tensor(y_train, dtype=torch.float32))
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, mode='regression')

    aggregated_importance = defaultdict(float)
    for i in range(len(X_test)):
        exp = explainer.explain_instance(X_test[i], predict_fn)
        feature_importance_dict = dict(exp.as_map()[1])
        for feature_idx, importance in feature_importance_dict.items():
            aggregated_importance[feature_names[feature_idx]] += abs(importance)

    for key in aggregated_importance.keys():
        aggregated_importance[key] /= len(X_test)

    print("\nAggregated Feature Importance (LIME - Across all test Dataset):")
    for feature, importance in sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")

    # Save feature importance to CSV
    dataset_name = os.path.basename(path).split(".")[0]
    importance_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Feature', 'Importance'])
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison
    csv_output_path = f"output/FI_Dataframes/LIME/{dataset_name}_lime_lstm_without_corr.csv"
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    importance_df.to_csv(csv_output_path, index=False)
    print(f"Feature importance saved successfully to: {csv_output_path}")

    plot_path = f"output/FI_Plots/LIME/{dataset_name}_lime_lstm_without_corr.png"
    print(f"Saving plot to: {plot_path}")
    plt.figure(figsize=(12, 6))
    plt.barh(list(aggregated_importance.keys()), list(aggregated_importance.values()), color='skyblue')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    plt.title('Aggregated Feature Importance (LIME - Across test Dataset)')
    for index, value in enumerate(aggregated_importance.values()):
        plt.text(value, index, f"{value:.4f}")
    plt.gca().invert_yaxis()
    plt.savefig(plot_path, format="png", dpi=300)
    print(f"Plot saved successfully: {os.path.exists(plot_path)}")
    plt.close()
    print(f"Processed and saved results for {dataset_name}")

    return {
        "test_loss": criterion(model(X_test), y_test).item(), 
        "top_features": top_features  
    }

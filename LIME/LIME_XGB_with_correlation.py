import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from collections import defaultdict
from prepare_data import prepare_data_with_correlation
import os

def process_dataset_with_lime_agg_xgb(path, prepare_data_func):
    X, y, feature_names = prepare_data_func(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='rmse')
    print("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    def predict_fn(input_data):
        return model.predict(input_data)

    explainer = LimeTabularExplainer(X_train, feature_names=feature_names, mode='regression')

    aggregated_importance = defaultdict(float)
    for i in range(len(X_test)):
        exp = explainer.explain_instance(X_test[i], predict_fn)
        feature_importance_dict = dict(exp.as_map()[1])
        for feature_idx, importance in feature_importance_dict.items():
            aggregated_importance[feature_names[feature_idx]] += abs(importance)

    for key in aggregated_importance.keys():
        aggregated_importance[key] /= len(X_test)

    print("\nAggregated Feature Importance (LIME - Across All Test Data):")
    for feature, importance in sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")

    # Save feature importance to CSV
    dataset_name = os.path.basename(path).split(".")[0]
    csv_output_path = f"output/FI_Dataframes/LIME/{dataset_name}_lime_xgb_withcorr.csv"
    importance_df = pd.DataFrame(list(aggregated_importance.items()), columns=['Feature', 'Importance'])
    importance_df.to_csv(csv_output_path, index=False)
    top_features = importance_df.head(10)["Feature"].tolist() #Added for comparison
    print(f"Feature importance saved successfully to: {csv_output_path}")

    plot_path = f"output/FI_Plots/LIME/{dataset_name}_lime_xgb_withcorr.png"
    print(f"Saving plot to: {plot_path}")
    plt.figure(figsize=(12, 6))
    plt.barh(list(aggregated_importance.keys()), list(aggregated_importance.values()), color='skyblue')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    plt.title('Aggregated Feature Importance (LIME with XGB - Across All Test Instances)')
    for index, value in enumerate(aggregated_importance.values()):
        plt.text(value, index, f"{value:.4f}")
    plt.gca().invert_yaxis()
    plt.savefig(plot_path, format="png", dpi=300)
    print(f"Plot saved successfully: {os.path.exists(plot_path)}")
    plt.close()
    print(f"Processed and saved results for {dataset_name}")

    return {
        "test_loss": rmse, 
        "top_features": top_features  
    }

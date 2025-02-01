import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from collections import defaultdict

def prepare_data():
    path = "static/Dataset/DMC2_AL_CP2.csv"
    df = pd.read_csv(path)

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

    correlation_matrix = X.corr()
    high_corr_features = [
        (col1, col2, correlation_matrix.loc[col1, col2])
        for col1 in correlation_matrix.columns
        for col2 in correlation_matrix.columns
        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
    ]
    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    retained_features = [f for f in all_features if f not in features_to_drop]
    X = X[retained_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, retained_features

X, y, feature_names = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='rmse')
print("Training XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

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

plt.figure(figsize=(12, 6))
plt.barh(list(aggregated_importance.keys()), list(aggregated_importance.values()), color='skyblue')
plt.xlabel('Average Importance')
plt.ylabel('Feature')
plt.title('Aggregated Feature Importance (LIME - Across All Test Instances)')
for index, value in enumerate(aggregated_importance.values()):
    plt.text(value, index, f"{value:.4f}")
plt.gca().invert_yaxis()
plt.show()

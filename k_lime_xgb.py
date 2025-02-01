import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load and preprocess the dataset with correlation handling
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
    y = df[targets].values

    # Compute correlation matrix and remove highly correlated features
    correlation_matrix = X.corr()
    threshold = 0.9  # Correlation threshold to remove highly correlated features

    correlated_features = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    # Drop correlated features
    X = X.drop(columns=correlated_features)
    retained_features = [col for col in all_features if col not in correlated_features]

    print(f"Removed correlated features: {correlated_features}")
    print(f"Remaining features: {retained_features}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, retained_features

# Prepare data
X, y, feature_names = prepare_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse'  # Specify evaluation metric here
)

print("Training XGBoost model...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

# Evaluate model
train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
print(f"Train R^2 Score: {train_score:.4f}")
print(f"Test R^2 Score: {test_score:.4f}")

# Apply LIME for feature importance
def predict_fn(input_data):
    return xgb_model.predict(input_data)

explainer = LimeTabularExplainer(X_train, feature_names=feature_names, mode='regression')

sample_idx = 0  # Select a test instance for explanation
exp = explainer.explain_instance(X_test[sample_idx], predict_fn)

# Extract feature importance
feature_importance_dict = dict(exp.as_map()[1])
feature_importance = {feature_names[i]: abs(importance) for i, importance in feature_importance_dict.items()}

# Print feature importance values
print("\nFeature Importance (LIME - XGB):")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(list(feature_importance.keys()), list(feature_importance.values()), color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (LIME-XGB)')
for index, value in enumerate(feature_importance.values()):
    plt.text(value, index, f"{value:.4f}")
plt.gca().invert_yaxis()
plt.show()

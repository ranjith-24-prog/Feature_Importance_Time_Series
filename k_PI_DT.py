import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset and preprocess it with correlation removal
def prepare_data_with_correlation_removal(file_path):
    df = pd.read_csv(file_path)

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

    # Find highly correlated feature pairs (correlation > 0.9)
    high_corr_features = set()
    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9:
                high_corr_features.add(col2)

    # Drop one feature from each highly correlated pair
    retained_features = [f for f in all_features if f not in high_corr_features]
    X = X[retained_features]

    return X, y, all_features, retained_features, correlation_matrix

# Provide the correct path to the CSV file
file_path = "static/Dataset/DMC2_AL_CP2.csv"

# Prepare the data
X, y, all_features, retained_features, correlation_matrix = prepare_data_with_correlation_removal(file_path)

# Display retained features and correlation matrix
print("\nRetained Features after Correlation Removal:")
print(retained_features)

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor for multi-output regression
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)

# Compute the Mean Squared Error for each target variable
mse_values = mean_squared_error(y_test, y_pred, multioutput='raw_values')
for i, target in enumerate(y.columns):
    print(f"Mean Squared Error for {target}: {mse_values[i]:.4f}")

# Apply Permutation Importance technique for multi-output
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)

# Display feature importance
print("\nFeature Importance (Permutation Importance):")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Permutation Importance)')
plt.gca().invert_yaxis()
plt.show()

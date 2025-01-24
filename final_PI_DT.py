import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from prepare_data_new import prepare_data_with_correlation_PI
import os


# Prepare the data
X, y, all_features, retained_features, correlation_matrix = prepare_data_with_correlation_PI()
# Display retained features and correlation matrix
print("\nRetained Features after Correlation Removal:")
print(retained_features)

print("\ncorrelation_matrix:")
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

# Save feature importance to CSV
csv_output_path = f"output/FI_Dataframes/Permutation/PI_DT_withcorr.csv"
importance_df.to_csv(csv_output_path, index=False)
print(f"Feature importance saved successfully to: {csv_output_path}")

# Display feature importance
print("\nFeature Importance (Permutation Importance):")
print(importance_df)

# Plot feature importance
plot_path = f"output/FI_Plots/Permutation/PI_DT_withcorr.png"
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Permutation Importance)')
plt.gca().invert_yaxis()
plt.savefig(plot_path, format="png", dpi=300)
print(f"Plot saved successfully: {os.path.exists(plot_path)}")
plt.show()

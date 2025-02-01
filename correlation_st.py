import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from st_aggrid import AgGrid, GridOptionsBuilder

# Function to get all features
def get_all_features():
    return [
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

# Function to prepare data and apply correlation filtering
def prepare_data_with_correlation(df, use_abs=True):
    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Compute correlation matrix
    correlation_matrix = X.corr()

    # Identify highly correlated features (ensuring unique pairs)
    high_corr_pairs = []
    seen_pairs = set()

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                pair = tuple(sorted([col1, col2]))  # Avoid duplicate pairs (A, B) == (B, A)
                if pair not in seen_pairs:
                    corr_value = correlation_matrix.loc[col1, col2]
                    if (abs(corr_value) if use_abs else corr_value) > 0.9:
                        high_corr_pairs.append((col1, col2, corr_value))
                        seen_pairs.add(pair)

    # Convert to DataFrame
    corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1', 'Feature 2', 'Correlation'])
    corr_df = corr_df.sort_values(by="Correlation", ascending=False)

    # Identify features to drop
    features_to_drop = list(set(pair[1] for pair in high_corr_pairs))
    retained_features = [f for f in all_features if f not in features_to_drop]
    X = X[retained_features]

    # Scale features and targets
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, retained_features, correlation_matrix, corr_df

# Function to plot the correlation matrix as a heatmap
def plot_correlation_matrix(correlation_matrix):
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# Function to display high correlations in an interactive AG-Grid
def display_correlation_aggrid(corr_df):
    st.write("### Highly Correlated Features (Interactive Table)")
    gb = GridOptionsBuilder.from_dataframe(corr_df)
    gb.configure_pagination()
    gb.configure_column("Correlation", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=4)
    grid_options = gb.build()
    AgGrid(corr_df, gridOptions=grid_options, height=300)

# Function to show features highly correlated with multiple others (with correlation values)
def get_multiple_high_corr_features(corr_df):
    feature_corr_map = {}

    # Build a dictionary of features and their correlated features with correlation values
    for _, row in corr_df.iterrows():
        feature_1, feature_2, corr_value = row['Feature 1'], row['Feature 2'], row['Correlation']

        # Store correlated features along with correlation value
        if feature_1 not in feature_corr_map:
            feature_corr_map[feature_1] = []
        if feature_2 not in feature_corr_map:
            feature_corr_map[feature_2] = []

        feature_corr_map[feature_1].append(f"{feature_2} ({corr_value:.4f})")
        feature_corr_map[feature_2].append(f"{feature_1} ({corr_value:.4f})")

    # Convert to DataFrame
    high_corr_df = pd.DataFrame(list(feature_corr_map.items()), columns=["Feature", "Highly Correlated Features"])
    high_corr_df["Highly Correlated Features"] = high_corr_df["Highly Correlated Features"].apply(lambda x: ", ".join(x))

    return high_corr_df

# Streamlit UI
st.title("Feature Correlation Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Toggle for using absolute correlation values
    use_abs_correlation = st.sidebar.checkbox("Use Absolute Correlation", value=True)

    # Data preprocessing
    X_scaled, y_scaled, retained_features, correlation_matrix, corr_df = prepare_data_with_correlation(df, use_abs=use_abs_correlation)

    # Sidebar options
    analysis_option = st.sidebar.radio(
        "Choose Analysis Type",
        [
            "Correlation Heatmap",
            "High Correlation Table",
            "Interactive Table",
            "Removed Features",
            "Highly Correlated Features",
            "Retained Features with Multiple Correlations",  # NEW OPTION
        ]
    )

    if analysis_option == "Correlation Heatmap":
        st.subheader("Correlation Matrix Heatmap")
        plot_correlation_matrix(correlation_matrix)

    elif analysis_option == "High Correlation Table":
        st.subheader("Highly Correlated Features Table")
        st.dataframe(corr_df)

    elif analysis_option == "Interactive Table":
        display_correlation_aggrid(corr_df)

    elif analysis_option == "Removed Features":
        st.write("### Removed Features due to High Correlation")
        removed_features = set(get_all_features()) - set(retained_features)
        st.write(removed_features)

        st.write("### Retained Features")
        st.write(retained_features)

    elif analysis_option == "Highly Correlated Features":
        st.subheader("Features Highly Correlated with Multiple Others (with Correlation Values)")
        high_corr_df = get_multiple_high_corr_features(corr_df)
        st.dataframe(high_corr_df)

    elif analysis_option == "Retained Features with Multiple Correlations":
        st.subheader("Multiple Correlations of Retained Features")

        # Filter `high_corr_df` to keep only retained features
        retained_corr_df = get_multiple_high_corr_features(corr_df)
        retained_corr_df = retained_corr_df[retained_corr_df["Feature"].isin(retained_features)]
        st.dataframe(retained_corr_df)

else:
    st.write("Please upload a CSV file to analyze the correlations.")

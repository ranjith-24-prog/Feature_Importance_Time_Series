import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Define all features and targets
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

def prepare_data_with_correlation(path):
    df = pd.read_csv(path)

    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Correlation filtering
    correlation_matrix = X.corr()
    high_corr_features = [
        (col1, col2)
        for col1 in correlation_matrix.columns
        for col2 in correlation_matrix.columns
        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
    ]
    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    retained_features = [f for f in all_features if f not in features_to_drop]
    print('Retained features with redundancy: ',retained_features)
    X = X[retained_features]

    # Scale features and targets
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, retained_features

def prepare_data_with_correlation_PI(path):
    df = pd.read_csv(path)

    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Correlation filtering
    correlation_matrix = X.corr()
    high_corr_features = [
        (col1, col2)
        for col1 in correlation_matrix.columns
        for col2 in correlation_matrix.columns
        if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.9
    ]
    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    retained_features = [f for f in all_features if f not in features_to_drop]
    X = X[retained_features]

    return  X, y, retained_features

def prepare_data_with_correlation_without_redundancy(path):
    df = pd.read_csv(path)

    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Correlation filtering
    correlation_matrix = X.corr()

    # Identify highly correlated features (ensuring unique pairs)
    high_corr_features = []
    seen_pairs = set()

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                pair = tuple(sorted([col1, col2]))  # Ensure unique order (A, B) == (B, A)
                if pair not in seen_pairs:
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.9:
                        high_corr_features.append((col1, col2))
                        seen_pairs.add(pair)

    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    retained_features = [f for f in all_features if f not in features_to_drop]
    print('Retained features wiithout redundancy: ',retained_features)
    X = X[retained_features]

    # Scale features and targets
    feature_scaler = StandardScaler()
    target_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    return X_scaled, y_scaled, retained_features

def prepare_data_with_correlation_without_redundancy_PI(path):
    df = pd.read_csv(path)

    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # Correlation filtering
    correlation_matrix = X.corr()

    # Identify highly correlated features (ensuring unique pairs)
    high_corr_features = []
    seen_pairs = set()

    for col1 in correlation_matrix.columns:
        for col2 in correlation_matrix.columns:
            if col1 != col2:
                pair = tuple(sorted([col1, col2]))  # Ensure unique order (A, B) == (B, A)
                if pair not in seen_pairs:
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.9:
                        high_corr_features.append((col1, col2))
                        seen_pairs.add(pair)

    features_to_drop = list(set(pair[1] for pair in high_corr_features))
    retained_features = [f for f in all_features if f not in features_to_drop]
    print('Retained features wiithout redundancy: ',retained_features)
    X = X[retained_features]

    return  X, y, retained_features


def prepare_data_without_correlation_x_y_scaled(path):
    df = pd.read_csv(path)

    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    return X_scaled, y_scaled, all_features

def prepare_data_without_correlation(path):
    df = pd.read_csv(path)

    # Feature and target selection (Keeping all features)
    all_features = get_all_features()
    targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    X = df[all_features]
    y = df[targets]

    # No correlation filtering (keeping all features)
    retained_features = all_features

    return X, y, retained_features

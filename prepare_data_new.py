import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data_with_correlation():
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

def prepare_data_without_correlation():
    path = "static/Dataset/DMC2_AL_CP2.csv"
    df = pd.read_csv(path)

    # Remove specific features and targets
    features_to_be_removed = [
        'CYCLE', 'A_DBD|0', 'POWER|1', 'POWER|2', 'POWER|3', 'POWER|4', 'POWER|5', 'POWER|6'
    ]

    # Add columns with '|4' and '|5' suffixes to the removal list
    features_to_be_removed.extend([col for col in df.columns if '|4' in col or '|5' in col])

    targets_to_be_removed = ['CURRENT|4', 'CURRENT|5']

    # Drop the columns
    columns_to_remove = features_to_be_removed + targets_to_be_removed
    df = df.drop(columns=columns_to_remove)

    # Define used features and targets
    used_features = [
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
        'ENC1_POS|1', 'ENC1_POS|2', 'ENC1_POS|3', 'ENC1_POS|6',
        'ENC2_POS|1', 'ENC2_POS|2', 'ENC2_POS|3', 'ENC2_POS|6'
    ]

    used_targets = ['CURRENT|1', 'CURRENT|2', 'CURRENT|3', 'CURRENT|6']

    # Extract features and targets
    features_data = df[used_features]
    target_data = df[used_targets]

    return features_data, target_data

def prepare_data_with_correlation_PI():
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

def get_all_features():
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
    return all_features
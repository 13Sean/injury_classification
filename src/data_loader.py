
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch


def categorize_apache(score):

    if score <= 2:
        return 0
    elif score <= 4:
        return 1
    elif score <= 6:
        return 2
    elif score <= 8:
        return 3
    else:
        return 4


def load_data(data_path, feature_names):

    data = pd.read_csv(data_path)
    features = data[feature_names].values
    target = np.array([categorize_apache(score) for score in data['Injury level'].values])
    return features, target


def normalize_features(features):

    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler


def split_data(features, target, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def convert_to_tensor(X_train, X_test, y_train, y_test, feature_dim):

    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, feature_dim)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, feature_dim)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test

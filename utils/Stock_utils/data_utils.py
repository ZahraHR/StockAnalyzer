import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler

def load_and_scale_data(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    features = ['Close']
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

    return feature_transform, scaler

def split_data(feature_transform: pd.DataFrame,
                train_ratio:float=0.6,
               val_ratio: float=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_index = int(len(feature_transform) * train_ratio)
    val_index = train_index + int(len(feature_transform) * val_ratio)

    X_train = feature_transform[:train_index]
    X_val = feature_transform[train_index:val_index]
    X_test = feature_transform[val_index:]

    return X_train, X_val, X_test

def create_sequence(dataset:pd.DataFrame, sequence_length: int=30) -> Tuple[np.ndarray, np.ndarray]:
    sequences, labels = [], []
    for start_idx in range(len(dataset) - sequence_length):
        stop_idx = start_idx + sequence_length
        sequences.append(dataset.iloc[start_idx:stop_idx].values)
        labels.append(dataset.iloc[stop_idx].values)

    return np.array(sequences), np.array(labels)

import torch

from torch.utils.data import Dataset
from torch.utils.data import Subset

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import fetch_openml


### Dataset Class

class CustomDataset(Dataset):
    def __init__(self, data, targets, indices, shuffle=True):
        if shuffle:
            perm = torch.randperm(len(data))
            self.data = data[perm]
            self.targets = targets[perm]
            self.indices = [indices[i] for i in perm.tolist()]
        else:
            self.data = data
            self.targets = targets
            self.indices = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Retrieve data and target using the index
        data_item = self.data[index]
        target_item = self.targets[index]
        index_item = self.indices[index]
        # Return data, target, and index
        return data_item, target_item, index_item


class IndexedSubset(Subset):
    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]
        return data, self.indices[idx]


def get_adult_dataset():
    # Fetch the Adult dataset
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame

    # Separate features (X) and target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Extract feature column names
    feature_names = X.columns.tolist()

    # Replace '?' with NaN and drop missing values
    X = X.replace('?', np.nan)
    X = X.dropna()
    y = y[X.index]  # Align target variable

    categorical_cols = X.select_dtypes(include=['category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    train_cols = categorical_cols.drop('sex')
    sensitive_cols = categorical_cols.drop(['workclass', 'education', 'marital-status', 'occupation',
                                            'relationship', 'race', 'native-country'])

    sex = X['sex']
    X['sex'] = (sex == 'Male').astype(int).values

    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False), train_cols)
        ])

    # Apply the preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Encode the target variable
    le = LabelEncoder()
    y_processed = le.fit_transform(y)

    # print(y_processed)

    X_dataset = torch.tensor(X_processed, dtype=torch.float32, device="cuda")
    y_dataset = torch.tensor(y_processed, dtype=torch.long, device="cuda")

    indices = X.index.tolist()

    return X, X_dataset, y_dataset, indices


def get_bankmarketing_dataset():
    # Fetch the Bank Marketing dataset
    bank = fetch_openml(name='bankmarketing', version=1, as_frame=True)
    df = bank.frame

    print(df)
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Replace 'unknown' with NaN and drop missing values
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].replace('unknown', np.nan)
    X = X.dropna()
    y = y[X.index]  # Align target variable

    # Extract sensitive attribute (age)
    age = X['age'].astype(float)
    X['age'] = ((age >= 25) & (age <= 60)).astype(int).values

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.drop('age')  # Exclude age from training

    train_cols = categorical_cols  # Use all other categorical columns
    # Drop 'age' from numerical features as it's our sensitive attribute
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False), train_cols)
        ])

    # Preprocess features
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Encode target labels
    le = LabelEncoder()
    y_processed = le.fit_transform(y)

    # Convert to PyTorch tensors
    X_dataset = torch.tensor(X_processed, dtype=torch.float32, device="cuda")
    y_dataset = torch.tensor(y_processed, dtype=torch.long, device="cuda")

    indices = X.index.tolist()

    return X, X_dataset, y_dataset, indices


def get_compass_dataset():
    # Fetch the Adult dataset
    compass = fetch_openml(data_id=44053, as_frame=True)
    df = compass.frame

    # Separate features (X) and target (y)
    X = df.drop('is_recid', axis=1)
    y = df['is_recid']

    # Extract feature column names
    feature_names = X.columns.tolist()

    # Replace '?' with NaN and drop missing values
    X = X.replace('?', np.nan)
    X = X.dropna()
    y = y[X.index]  # Align target variable

    categorical_cols = X.select_dtypes(include=['category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    train_cols = categorical_cols.drop('race')

    race = X['race'].astype(int)
    X_filtered = X[(race == 0) | (race == 2)].copy()  # make an explicit copy

    X_filtered['race'] = (X_filtered['race'].astype(int) == 0).astype(int)

    X = X_filtered


    # Define the preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False), train_cols)
        ])

    # Apply the preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Encode the target variable
    le = LabelEncoder()
    y_processed = le.fit_transform(y)

    # print(y_processed)

    X_dataset = torch.tensor(X_processed, dtype=torch.float32, device="cuda")
    y_dataset = torch.tensor(y_processed, dtype=torch.long, device="cuda")

    indices = X.index.tolist()

    return X, X_dataset, y_dataset, indices


def get_dutch_dataset():
    df = pd.read_csv('./dutch.csv')
    
    # Convert 'sex' to 'Male'/'Female' (if needed) and binary format (1 = Male, 0 = Female)
    df['sex'] = df['sex'].apply(lambda v: 'Male' if v.lower() == 'male' else 'Female')
    df['sex'] = (df['sex'] == 'Male').astype(int)

    protected_attribute = 'sex'
    majority_group_name = 'Male'
    minority_group_name = 'Female'
    class_label = 'occupation'

    # Label encode all categorical variables
    le = preprocessing.LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Separate features and target
    X = df.drop(columns=[class_label])
    y = df[class_label]

    # Extract categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['int64']).columns.drop(protected_attribute)
    numerical_cols = X.select_dtypes(include=['float64']).columns

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
        ]
    )

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    # Encode the target variable
    y_processed = le.fit_transform(y)

    # Convert to tensors
    X_dataset = torch.tensor(X_processed, dtype=torch.float32, device="cuda")
    y_dataset = torch.tensor(y_processed, dtype=torch.long, device="cuda")

    indices = X.index.tolist()

    return X, X_dataset, y_dataset, indices

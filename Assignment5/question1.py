from sklearn.model_selection import train_test_split
import numpy as np

DATASETS = ["datasets/mnist_train_data.npy", "datasets/mnist_train_labels.npy"]

def data_split():
    train_data = np.load(DATASETS[0])
    test_labels = np.load(DATASETS[1])
    print(f"Number of data items: {len(train_data)}")
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, test_labels, test_size=.1, random_state=0, stratify=test_labels
    )
    print(f"Split into {len(X_train)} items for training set ({len(X_train)/len(train_data) * 100}% of the complete data set)")
    print(f"Split into {len(X_test)} items for test set ({len(X_test)/len(train_data) * 100}% of the complete data set)")
    print("Stratify option in \'train_test_split\' function helps with a balanced outcome with the split")
    return X_train, X_test, y_train, y_test

# data_split() For testing purposes
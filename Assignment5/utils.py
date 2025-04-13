import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

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

def img_to_arr(dataset_imgs):
    arr_form = []
    for item in dataset_imgs:
        arr_form.append(item.flatten(order='F')) # Column-major (Fortran-style) order. Elements are read in column-by-column
    return arr_form

# Evaluate model
def evaluate_model(model, X_test, y_test, model_name, mode):
    print(f"\n---------Evaluating {model_name} model-------------")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=mode)
    recall = recall_score(y_test, y_pred, average=mode)
    f1 = f1_score(y_test, y_pred, average=mode)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
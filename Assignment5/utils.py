import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

def convert_img_to_arr(dataset_imgs):
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
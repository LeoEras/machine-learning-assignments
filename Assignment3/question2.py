import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from features_helper import read_features

# List of 14 brain region features
FEATURES = read_features('datasets/fdg_pet.feature.info.txt')
C_SEARCH = np.arange(3, 5, 0.1) # Always centers somewhere around 4

# Step 1: Load and Prepare Data (Fix Column Header Issue)
def load_data():
    # Load training data
    train_nc = pd.read_csv("datasets/train.fdg_pet.sNC.csv", header=None, names=FEATURES)
    train_dat = pd.read_csv("datasets/train.fdg_pet.sDAT.csv", header=None, names=FEATURES)

    # Load test data
    test_nc = pd.read_csv("datasets/test.fdg_pet.sNC.csv", header=None, names=FEATURES)
    test_dat = pd.read_csv("datasets/test.fdg_pet.sDAT.csv", header=None, names=FEATURES)

    # Assign labels (0 = sNC, 1 = sDAT)
    train_nc["label"] = 0
    train_dat["label"] = 1
    test_nc["label"] = 0
    test_dat["label"] = 1

    # Merge train and test data
    train_data = pd.concat([train_nc, train_dat], ignore_index=True)
    test_data = pd.concat([test_nc, test_dat], ignore_index=True)

    # Separate features and labels
    X_train = train_data.drop(columns=["label"])
    y_train = train_data["label"]
    X_test = test_data.drop(columns=["label"])
    y_test = test_data["label"]

    return X_train, y_train, X_test, y_test

# Step 2: Perform Grid Search to Find Best '(C, d)'
def train_polynomial_svm(X_train, y_train, use_subset=False):
    print("\nTraining Polynomial SVM using Grid Search...")

    # Reduce dataset size for faster testing (Optional)
    if use_subset:
        X_train = X_train.sample(frac=0.5, random_state=42)
        y_train = y_train.loc[X_train.index]

    param_grid = {
        "C": C_SEARCH,  # Fewer C values for speed
        "degree": [2, 3, 4]  # Removed degree 5 for speed
    }
    svm = SVC(kernel="poly")

    # Cross-validation grid search (Reduced cv=3 for speed)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy", verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_C = best_params["C"]
    best_degree = best_params["degree"]
    print(f"Best parameters found: C = {best_C}, Degree = {best_degree}")

    return best_C, best_degree, grid_search.cv_results_

# Step 3: Train Final Model with Best '(C, d)'
def train_final_model(X_train, y_train, best_C, best_degree):
    print("\nTraining final Polynomial SVM model with best parameters...")
    final_svm = SVC(kernel="poly", C=best_C, degree=best_degree)
    final_svm.fit(X_train, y_train)
    print("Final model training complete!")
    return final_svm

# Step 4: Evaluate Model Performance
def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Compute confusion matrix for specificity calculation
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, _, _ = cm.ravel()
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2

    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, specificity, balanced_accuracy

# Step 5: Plot Model Performance for Different '(C, d)' Values
def plot_C_performance(cv_results):
    # C_values = np.logspace(-3, 3, 5)
    degrees = [2, 3, 4]

    mean_scores = cv_results["mean_test_score"].reshape(len(C_SEARCH), len(degrees))

    plt.figure(figsize=(8, 6))
    for i, d in enumerate(degrees):
        plt.plot(C_SEARCH, mean_scores[:, i], marker="o", linestyle="-", label=f"Degree {d}")

    #plt.xscale("log")
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Mean Accuracy")
    plt.title("Polynomial SVM Performance for Different (C, d)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":  
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Find best '(C, d)' using Grid Search
    best_C, best_degree, cv_results = train_polynomial_svm(X_train, y_train, use_subset=True)

    # Train final SVM with best parameters
    final_model = train_final_model(X_train, y_train, best_C, best_degree)

    # Evaluate the model
    evaluate_model(final_model, X_test, y_test)

    # Plot performance of different '(C, d)' values
    plot_C_performance(cv_results)

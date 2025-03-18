import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from utils import load_data

C_SEARCH = np.logspace(-3, 3, 5)
GAMMA_SEARCH = np.logspace(-3, 3, 5)

def train_rbf_svm(X_train, y_train):
    print("\nTraining RBF SVM using Grid Search...")

    param_grid = {
        "C": C_SEARCH,  # Fewer C values for speed
        "gamma": GAMMA_SEARCH  # γ values to tune
    }
    svm = SVC(kernel="rbf")

    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring="accuracy", verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_C = best_params["C"]
    best_gamma = best_params["gamma"]
    print(f"Best parameters found: C = {best_C}, Gamma = {best_gamma}")

    return best_C, best_gamma, grid_search.cv_results_

def train_final_model(X_train, y_train, best_C, best_gamma):
    print("\nTraining final RBF SVM model with best parameters...")
    final_svm = SVC(kernel="rbf", C=best_C, gamma=best_gamma)
    final_svm.fit(X_train, y_train)
    print("Final model training complete!")
    return final_svm

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
    tn, fp,  _, _= cm.ravel()
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

def plot_C_performance(cv_results):
    mean_scores = cv_results["mean_test_score"].reshape(len(C_SEARCH), len(GAMMA_SEARCH))

    plt.figure(figsize=(8, 6))
    for i, g in enumerate(GAMMA_SEARCH):
        plt.plot(C_SEARCH, mean_scores[:, i], marker="o", linestyle="-", label=f"Gamma {g}")

    plt.xscale("log")
    plt.xlabel("C Value (Log Scale)")
    plt.ylabel("Mean Accuracy")
    plt.title("RBF SVM Performance for Different (C, gamma)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Find best '(C, gamma)' using Grid Search
    best_C, best_gamma, cv_results = train_rbf_svm(X_train, y_train)

    # Train final SVM with best parameters
    final_model = train_final_model(X_train, y_train, best_C, best_gamma)

    # Evaluate the model
    evaluate_model(final_model, X_test, y_test)

    # Plot performance of different '(C, gamma)' values
    plot_C_performance(cv_results)

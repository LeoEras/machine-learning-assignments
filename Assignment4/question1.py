from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
from utils import load_data

warnings.filterwarnings("ignore")

# Train Decision Tree with Grid Search
def train_decision_tree(X_train, y_train):
    print("Running Grid Search on Decision Tree criteria...")
    param_grid = {"criterion": ["gini", "entropy", "log_loss"]}
    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    best_criterion = grid_search.best_params_["criterion"]
    print(f"Best criterion: {best_criterion}")
    return best_criterion

# Train final model with best criterion
def train_final_model(X_train, y_train, best_criterion):
    print("Training final decision tree")
    model = DecisionTreeClassifier(criterion=best_criterion, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, _, _ = cm.ravel()
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Main execution
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    best_criterion = train_decision_tree(X_train, y_train)
    model = train_final_model(X_train, y_train, best_criterion)
    evaluate_model(model, X_test, y_test)

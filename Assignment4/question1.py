from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import warnings
from utils import load_data, evaluate_model

warnings.filterwarnings("ignore")

# Train Decision Tree with Grid Search
def train_decision_tree(X_train, y_train):
    print("Running Grid Search on Decision Tree criteria...")
    param_grid = {"criterion": ["gini", "entropy", "log_loss"]}
    # A convention, using the number 42
    tree = DecisionTreeClassifier(random_state=42) # Answer to the Ultimate Question of Life, the Universe, and Everything.
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

# Main execution
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    best_criterion = train_decision_tree(X_train, y_train)
    model = train_final_model(X_train, y_train, best_criterion)
    evaluate_model(model, X_test, y_test, "Decision Tree")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils import load_data, evaluate_model

# Train Decision Tree with Grid Search
def train_random_forest(X_train, y_train):
    print("Running Grid Search on Random Forest criteria...")
    param_grid = {"criterion": ["gini", "entropy", "log_loss"]}
    # A convention, using the number 42
    tree = RandomForestClassifier(random_state=42) # Answer to the Ultimate Question of Life, the Universe, and Everything.
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    best_criterion = grid_search.best_params_["criterion"]
    print(f"Best criterion: {best_criterion}")
    return best_criterion

# Train final model with best criterion
def train_final_model(X_train, y_train, best_criterion):
    model = RandomForestClassifier(criterion=best_criterion, random_state=42)
    model.fit(X_train, y_train)
    return model

# Main execution
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    best_criterion = train_random_forest(X_train, y_train)
    forest_model = train_final_model(X_train, y_train, best_criterion)
    dt_model = DecisionTreeClassifier(criterion="gini", random_state=42)
    dt_model.fit(X_train, y_train)
    evaluate_model(forest_model, X_test, y_test, "Random Forest")
    evaluate_model(dt_model, X_test, y_test, "Decision Tree")

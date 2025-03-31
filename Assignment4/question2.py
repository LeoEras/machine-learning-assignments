from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils import load_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("This program saves the tree as a local .png file, as these are too big to be seen by human eyes (matplotlib)")
    X_train, y_train, X_test, y_test = load_data()
    model = DecisionTreeClassifier(criterion="gini", random_state=42)
    model.fit(X_train, y_train)
    plt.figure(figsize=(40, 10))
    plot_tree(model, fontsize=7, max_depth=4, feature_names=X_train.columns, filled=True)
    plt.savefig("tree_level3.png", bbox_inches='tight')

    plt.figure(figsize=(70, 30))
    plot_tree(model, fontsize=7, feature_names=X_train.columns, filled=True)
    plt.savefig("tree_complete.png", bbox_inches='tight')

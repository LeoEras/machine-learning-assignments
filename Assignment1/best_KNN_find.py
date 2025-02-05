import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_sDAT = pd.read_csv("datasets/train.sDAT.csv", header=None)
train_sNC = pd.read_csv("datasets/train.sNC.csv", header=None)
test_sDAT = pd.read_csv("datasets/test.sDAT.csv", header=None)
test_sNC = pd.read_csv("datasets/test.sNC.csv", header=None)

# Data cleaning (Training data)
y_train_sDAT = pd.Series([1 for _ in range(len(train_sDAT))])
y_train_sNC = pd.Series([0 for _ in range(len(train_sNC))])
x_train_merged = pd.concat([train_sDAT, train_sNC], ignore_index=True, sort=False)
y_train_merged = pd.concat([y_train_sDAT, y_train_sNC], ignore_index=True, sort=False)

# Data cleaning (Testing data)
y_test_sDAT = pd.Series([1 for _ in range(len(test_sDAT))])
y_test_sNC = pd.Series([0 for _ in range(len(test_sNC))])
x_test_merged = pd.concat([test_sDAT, test_sNC], ignore_index=True, sort=False)
y_test_merged = pd.concat([y_test_sDAT, y_test_sNC], ignore_index=True, sort=False)


k_values_to_test = range(1, 200)
dist_metrics = ['russellrao', 'l1', 'cosine', 'l2', 'p', 'canberra', 'sokalmichener', 'infinity', 'jaccard', 'rogerstanimoto', 'euclidean', 'yule', 'hamming', 'nan_euclidean', 'cityblock', 'manhattan', 'chebyshev', 'minkowski', 'dice', 'haversine', 'correlation', 'sokalsneath', 'braycurtis']
warnings.filterwarnings("ignore")

for dist_m in dist_metrics:
    acc = 0
    k_val = 0
    for k_value in k_values_to_test:
        try:
            knn_model = KNeighborsClassifier(n_neighbors=k_value, metric=dist_m)
            knn_model.fit(x_train_merged, y_train_merged)

            # Calculate the accuracy of the model.
            y_test_pred = knn_model.predict(x_test_merged)
            test_accuracy = accuracy_score(y_test_merged, y_test_pred)
            y_train_pred = knn_model.predict(x_train_merged)
            train_accuracy = accuracy_score(y_train_merged, y_train_pred)

            if test_accuracy > acc:
                acc = test_accuracy
                k_val = k_value
        except:
            print("Something went wrong with " + dist_m)

    print("Best values for KNN using " + dist_m + " metric: " + str(acc) + " with " + str(k_val))


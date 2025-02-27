import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# Reading the data
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

x_dataset = train_df[train_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
y_labels = train_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]

# First split -> Blob x, x_test, blob y, y_test
# Using 80% for blob, 20% for test
blob_x, x_test, blob_y, y_test = train_test_split(x_dataset, y_labels, test_size=0.2, train_size=0.8)
# Second split -> train x, x_validation, train y, y_validation
# Using 75% for train (75% of blob), 25% for test (25% of the blob)
# End up with something like 60%, 20%, 20% for train, validation and test respectively
x_train, x_cv, y_train, y_cv = train_test_split(blob_x, blob_y, test_size = 0.25, train_size =0.75)

train_validation_test_regressor = LinearRegression().fit(blob_x, blob_y) # Blobs are already train + validation
y_pred = train_validation_test_regressor.predict(x_test)

print("USING THE TRAIN, VALIDATION AND TEST APPROACH")
# TRAIN - VALIDATION - TEST SPLIT IS RANDOM, SO RANDOM RESULTS HERE, AS SEEN IN CLASS
print(f"Residual Standard Error (RSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred):.2f}")

# Using cross-validation (k = 5, then k = 10)
# This should perform worse as K is large (compared to k = 5)
k_values = [5, 10]
for k_val in k_values:
    print("\nUSING THE CROSS VALIDATION APPROACH, K = " + str(k_val))
    k_folds = KFold(n_splits = k_val)
    scores = cross_val_score(LinearRegression(), x_dataset, y_labels, cv = k_folds, scoring='neg_mean_squared_error')
    # Convert negative mean squared error to positive and take the square root to get RSE
    cv_rse = np.sqrt(-scores.mean())
    # Calculate R-squared for cross-validation
    cv_r2 = cross_val_score(LinearRegression(), x_dataset, y_labels, cv = k_folds, scoring='r2').mean()
    print(f"Residual Standard Error (RSE): {cv_rse:.2f}")
    print(f"R-squared (R2): {cv_r2:.2f}")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# Reading the data
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

x_dataset = train_df[train_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
y_labels = train_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]
x_test = test_df[test_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
y_test = test_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]

# Tested with np.arange(1, 10000, 0.001), bad idea, it took ages
# One thing to notice is that
# np.arange(0.01, 10, 0.01)  -  small 0.01 increments, always yielded the largest number (closer to 10)
# So, I tested some values, first 10, then 100 (starting from 10), then 1000 (starting from 100)
# After some time, I got the value settles near 5015 (5015.20 seems to be good enough)
alpha_list = np.arange(5000, 5050, 0.1) # 0.1, 0.01 takes too much time
alpha_space = {'alpha': alpha_list }

# Using 5 instead of 10, as we saw on question 1 it performed slighly better
ridge_grid = GridSearchCV(Ridge(), alpha_space, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(x_dataset, y_labels)

best_alpha = ridge_grid.best_params_["alpha"]
rmse_values = np.sqrt(-ridge_grid.cv_results_['mean_test_score']) # For graphics purposes

ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(x_dataset, y_labels)
y_pred = ridge_model.predict(x_test)

print("Ridge results")
print(f"Residual Standard Error (RSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred):.2f}")

# I assume we are using the best one from question 1 to compare
print("\nBest one from Question 1 (CV - 5 fold)")
k_folds = KFold(n_splits = 5)
scores = cross_val_score(LinearRegression(), x_dataset, y_labels, cv = k_folds, scoring='neg_mean_squared_error')
# Convert negative mean squared error to positive and take the square root to get RSE
cv_rse = np.sqrt(-scores.mean())
# Calculate R-squared for cross-validation
cv_r2 = cross_val_score(LinearRegression(), x_dataset, y_labels, cv = k_folds, scoring='r2').mean()
print(f"Residual Standard Error (RSE): {cv_rse:.2f}")
print(f"R-squared (R2): {cv_r2:.2f}")

plt.plot(alpha_list, rmse_values)
plt.xlabel('Alpha values')
plt.ylabel('Cross-Validation RMSE')
plt.title('Ridge Regression: Alpha effect on RSM')
plt.show()
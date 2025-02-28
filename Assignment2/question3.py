import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

# Reading the data
train_df = pd.read_csv("datasets/train.csv")
test_df = pd.read_csv("datasets/test.csv")

x_dataset = train_df[train_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
y_labels = train_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]
x_test = test_df[test_df.columns.drop("ConcreteCompressiveStrength_MPa_Megapascals_")]
y_test = test_df[["ConcreteCompressiveStrength_MPa_Megapascals_"]]

# Seems to be very close to zero this time, as testing between 1 and 1000 gave the number 1
alpha_list = np.arange(0.001, 0.04, 0.001)
alpha_space = {'alpha': alpha_list }
lasso = Lasso()

lasso_grid = GridSearchCV(lasso, alpha_space, scoring='neg_mean_squared_error', cv=5)
lasso_grid.fit(x_dataset, y_labels)

best_alpha = lasso_grid.best_params_["alpha"]
rmse_values = np.sqrt(-lasso_grid.cv_results_['mean_test_score']) # For graphics purposes

lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(x_dataset, y_labels)
y_pred = lasso_model.predict(x_test)

print("Lasso results")
print(f"Residual Standard Error (RSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred):.4f}")

# Ridge model
print("\nRidge model, best from question 2")
ridge_model = Ridge(alpha=5015.2) # Best alpha from question 2
ridge_model.fit(x_dataset, y_labels)
y_pred = ridge_model.predict(x_test)
print(f"Residual Standard Error (RSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R-squared (R2): {r2_score(y_test, y_pred):.4f}")

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
plt.title('Lasso Regression: Alpha effect on RSM')
plt.show()
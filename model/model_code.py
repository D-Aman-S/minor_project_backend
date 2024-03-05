import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np
# import matplotlib.pyplot as plt
import pickle

# Load data from Excel file
data = pd.read_excel("C:\\Users\\amans\\Desktop\\minor project\\minor_project.xlsx")

# Drop rows with null values in 'nu_exp' column
data = data.dropna(subset=['nu_exp'])

# Split data into independent and dependent variables
X = data[['ja_sup', 'ja_sub', 'Re_V', 'Re_L', 'prv']]
y = data['nu_exp']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=38)

# Define models and hyperparameters for grid search
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
}
parameters = {
    'Linear Regression': {},
    'Ridge': {'alpha': [0.1, 1, 10]},
    'Lasso': {'alpha': [0.1, 1, 10]},
    'Elastic Net': {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 7]},
}

# Perform grid search with cross-validation
best_model = None
best_score = -np.inf
for name, model in models.items():
    grid_search = GridSearchCV(model, parameters[name], cv=5)
    grid_search.fit(X_train, y_train)
    score = grid_search.score(X_test, y_test)
    print(f"{name}: Best parameters {grid_search.best_params_}, Best score {score}")
    if score > best_score:
        best_model = grid_search.best_estimator_
        best_score = score

# Use the best model for predictions
predictions = best_model.predict(X_test)
error = np.mean((predictions - y_test) ** 2)
accuracy = best_model.score(X_test, y_test)

#svaing the model
pickle.dump(best_model,open('model/gradientbooster.pkl','wb'))

print("Best Model:")
print("  Model:", type(best_model).__name__)
print("  Mean Squared Error:", error)
print("  Accuracy:", accuracy)

# Calculate the mean of nu_mod
mean_nu_mod = y_test.mean()

# Calculate the error percentage
error_percentage = (error / mean_nu_mod) * 100
print("Error Percentage:", error_percentage)

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, predictions, color='blue', label='Predictions')
# plt.plot(y_test, y_test, color='red', linestyle='--', label='Actual')
# plt.title('Regression Predictions vs Actual')
# plt.xlabel('Actual Values')
# plt.ylabel('Predicted Values')
# plt.legend()
# plt.grid(True)
# plt.show()

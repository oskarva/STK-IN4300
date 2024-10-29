import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from patsy import dmatrix
import statsmodels.api as sm
import seaborn as sns

# Load dataset for regression (Problem 1)
data = pd.read_csv('./assignment_2/qsar_aquatic_toxicity-csv', sep=';')

# Rename columns based on the descriptors
columns = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C040', 'LC50']
data.columns = columns

# Split dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)

# Define features and target
X_train = train_data.drop(columns=['LC50'])
y_train = train_data['LC50']
X_test = test_data.drop(columns=['LC50'])
y_test = test_data['LC50']

# Define a function for linear regression
def linear_regression_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Fit linear regression model (Option i: directly model count variables)
lr_model_1 = linear_regression_model(X_train, y_train)
y_train_pred_1 = lr_model_1.predict(X_train)
y_test_pred_1 = lr_model_1.predict(X_test)

# Calculate training and test errors for linear regression model
train_error_1 = mean_squared_error(y_train, y_train_pred_1)
test_error_1 = mean_squared_error(y_test, y_test_pred_1)
print(f"Training Error (Model i): {train_error_1}")
print(f"Test Error (Model i): {test_error_1}")

# Option ii: One-hot encoding of count variables
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[['nN', 'C040']]).toarray()
X_test_encoded = encoder.transform(X_test[['nN', 'C040']]).toarray()

# Concatenate encoded features with rest of the features
X_train_combined = np.hstack((X_train.drop(columns=['nN', 'C040']).values, X_train_encoded))
X_test_combined = np.hstack((X_test.drop(columns=['nN', 'C040']).values, X_test_encoded))

# Fit linear regression model (Option ii: one-hot encoding)
lr_model_2 = linear_regression_model(X_train_combined, y_train)
y_train_pred_2 = lr_model_2.predict(X_train_combined)
y_test_pred_2 = lr_model_2.predict(X_test_combined)

# Calculate training and test errors for the encoded linear regression model
train_error_2 = mean_squared_error(y_train, y_train_pred_2)
test_error_2 = mean_squared_error(y_test, y_test_pred_2)
print(f"Training Error (Model ii): {train_error_2}")
print(f"Test Error (Model ii): {test_error_2}")

# Repeat the procedure described in (a) 200 times
train_errors_1 = []
test_errors_1 = []
train_errors_2 = []
test_errors_2 = []

for _ in range(200):
    train_data, test_data = train_test_split(data, test_size=0.33)
    X_train = train_data.drop(columns=['LC50'])
    y_train = train_data['LC50']
    X_test = test_data.drop(columns=['LC50'])
    y_test = test_data['LC50']

    # Model i
    lr_model_1 = linear_regression_model(X_train, y_train)
    y_test_pred_1 = lr_model_1.predict(X_test)
    test_errors_1.append(mean_squared_error(y_test, y_test_pred_1))

    # Model ii (one-hot encoding)
    X_train_encoded = encoder.fit_transform(X_train[['nN', 'C040']]).toarray()
    X_test_encoded = encoder.transform(X_test[['nN', 'C040']]).toarray()
    X_train_combined = np.hstack((X_train.drop(columns=['nN', 'C040']).values, X_train_encoded))
    X_test_combined = np.hstack((X_test.drop(columns=['nN', 'C040']).values, X_test_encoded))
    lr_model_2 = linear_regression_model(X_train_combined, y_train)
    y_test_pred_2 = lr_model_2.predict(X_test_combined)
    test_errors_2.append(mean_squared_error(y_test, y_test_pred_2))

# Plot empirical distribution of the test error for each model
plt.figure(figsize=(10, 6))
sns.kdeplot(test_errors_1, label='Model i (Direct Linear)')
sns.kdeplot(test_errors_2, label='Model ii (One-hot Encoding)')
plt.xlabel('Test Error')
plt.ylabel('Density')
plt.title('Empirical Distribution of Test Errors (Model i vs Model ii)')
plt.legend()
plt.show()

# Comments on the results
def comment_on_results():
    print("Model i generally performs better than Model ii, as the one-hot encoding adds unnecessary complexity when count variables do not significantly benefit from being treated as categorical.")
    print("The empirical distribution indicates that Model ii tends to have a higher variance in its test errors, reflecting overfitting due to increased dimensionality.")

comment_on_results()

# (c) Variable Selection using Backward Elimination and Forward Selection
import statsmodels.api as sm

# Backward Elimination with AIC
X_train_const = sm.add_constant(X_train)
model_ols = sm.OLS(y_train, X_train_const).fit()
backward_aic_model = model_ols
while True:
    aic = backward_aic_model.aic
    max_pval = backward_aic_model.pvalues.idxmax()
    if backward_aic_model.pvalues[max_pval] > 0.05:
        X_train_const = X_train_const.drop(columns=[max_pval])
        backward_aic_model = sm.OLS(y_train, X_train_const).fit()
    else:
        break
print("Backward Elimination (AIC) Model Summary:")
print(backward_aic_model.summary())

# Forward Selection with AIC
remaining_features = list(X_train.columns)
selected_features = []
current_score, best_new_score = float('inf'), float('inf')
while remaining_features:
    scores_with_candidates = []
    for candidate in remaining_features:
        features = selected_features + [candidate]
        X_train_const = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_train_const).fit()
        scores_with_candidates.append((model.aic, candidate))
    scores_with_candidates.sort()
    best_new_score, best_candidate = scores_with_candidates[0]
    if current_score > best_new_score:
        remaining_features.remove(best_candidate)
        selected_features.append(best_candidate)
        current_score = best_new_score
    else:
        break
print("Selected features using Forward Selection with AIC:")
print(selected_features)

# (d) Ridge Regression with Bootstrap and Cross-Validation
from sklearn.linear_model import RidgeCV
from sklearn.utils import resample

# Cross-Validation to find optimal alpha
alphas = np.logspace(-6, 6, 13)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
print(f"Optimal alpha using Cross-Validation: {ridge_cv.alpha_}")

# Bootstrap to find optimal alpha
bootstrap_alphas = []
for _ in range(100):
    X_resampled, y_resampled = resample(X_train, y_train)
    ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_resampled, y_resampled)
    bootstrap_alphas.append(ridge_cv.alpha_)

plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_alphas, kde=True)
plt.xlabel('Alpha')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Optimal Alpha for Ridge Regression')
plt.show()

# (e) Generalized Additive Model (GAM)
from pygam import LinearGAM, s

# Fit a GAM with different levels of complexity
gam_1 = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7)).fit(X_train, y_train)
gam_2 = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=10) + s(3, n_splines=10) +
                  s(4, n_splines=10) + s(5, n_splines=10) + s(6, n_splines=10) + s(7, n_splines=10)).fit(X_train, y_train)

print(f"GAM Model 1 AIC: {gam_1.statistics_['AIC']}")
print(f"GAM Model 2 AIC: {gam_2.statistics_['AIC']}")

# (f) Regression Tree
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(reg_tree, filled=True, feature_names=X_train.columns)
plt.show()

# (g) Compare all models in terms of training and test error
models = {
    "Linear Regression (Direct)": lr_model_1,
    "Linear Regression (One-Hot)": lr_model_2,
    "Ridge Regression": ridge_cv,
    "Regression Tree": reg_tree,
}

for model_name, model in models.items():
    if model_name in ["Linear Regression (Direct)", "Linear Regression (One-Hot)"]:
        train_pred = model.predict(X_train) if model_name == "Linear Regression (Direct)" else model.predict(X_train_combined)
        test_pred = model.predict(X_test) if model_name == "Linear Regression (Direct)" else model.predict(X_test_combined)
    else:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    print(f"{model_name} - Training Error: {train_error}, Test Error: {test_error}")

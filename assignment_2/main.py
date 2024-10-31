import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from patsy import dmatrix
import statsmodels.api as sm
import seaborn as sns
import random
from pygam import LinearGAM, LogisticGAM, s

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Load dataset for regression (Problem 1)
data = pd.read_csv('./assignment_2/qsar_aquatic_toxicity-csv', sep=';')

# Rename columns based on the descriptors
columns = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI', 'GATS1p', 'nN', 'C040', 'LC50']
data.columns = columns

# Split dataset into training and test sets
train_data, test_data = train_test_split(data, test_size=0.33)

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

# Regression coefficients for Model i
print("Regression Coefficients (Model i):")
print(lr_model_1.coef_)
print("Intercept (Model i):")
print(lr_model_1.intercept_)

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

# Regression coefficients for Model ii
print("Regression Coefficients (Model ii):")
print(lr_model_2.coef_)
print("Intercept (Model ii):")
print(lr_model_2.intercept_)

# Extract regression coefficients for comparison
common_features = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI', 'GATS1p']
coefficients_model_i = lr_model_1.coef_[:len(common_features)]
coefficients_model_ii = lr_model_2.coef_[:len(common_features)]

# Generate LaTeX table for common coefficients
table = ""
for feature, coef_i, coef_ii in zip(common_features, coefficients_model_i, coefficients_model_ii):
    table += f"{feature} & {coef_i:.4f} & {coef_ii:.4f} \\\n"


print(table)


# Repeat the procedure described in (a) 200 times
train_errors_1 = []
test_errors_1 = []
train_errors_2 = []
test_errors_2 = []

for i in range(200):
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=i)
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
plt.savefig(f'plot_output_{random.randint(0, 10000)}.png', format='png', dpi=300)
plt.close()


# (c) Variable Selection using Backward Elimination and Forward Selection
import statsmodels.api as sm

# Backward Elimination with AIC
X_train_const = sm.add_constant(X_train)
model_ols = sm.OLS(y_train, X_train_const).fit()
backward_aic_model = model_ols
while True:
    current_aic = backward_aic_model.aic
    max_pval = backward_aic_model.pvalues.idxmax()
    if backward_aic_model.pvalues[max_pval] > 0.05:
        X_train_const = X_train_const.drop(columns=[max_pval])
        new_model = sm.OLS(y_train, X_train_const).fit()
        if new_model.aic < current_aic:
            backward_aic_model = new_model
        else:
            break
    else:
        break
print("Backward Elimination (AIC) Model Summary:")
print(backward_aic_model.summary())

# Backward Elimination with BIC
X_train_const_bic = sm.add_constant(X_train)
model_ols_bic = sm.OLS(y_train, X_train_const_bic).fit()
backward_bic_model = model_ols_bic
while True:
    current_bic = backward_bic_model.bic
    max_pval = backward_bic_model.pvalues.idxmax()
    if backward_bic_model.pvalues[max_pval] > 0.05:
        X_train_const_bic = X_train_const_bic.drop(columns=[max_pval])
        new_model_bic = sm.OLS(y_train, X_train_const_bic).fit()
        if new_model_bic.bic < current_bic:
            backward_bic_model = new_model_bic
        else:
            break
    else:
        break
print("Backward Elimination (BIC) Model Summary:")
print(backward_bic_model.summary())

# Forward Selection with AIC
remaining_features = list(X_train.columns)
selected_features_aic = []
current_score, best_new_score = float('inf'), float('inf')
while remaining_features:
    scores_with_candidates = []
    for candidate in remaining_features:
        features = selected_features_aic + [candidate]
        X_train_const = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_train_const).fit()
        scores_with_candidates.append((model.aic, candidate))
    scores_with_candidates.sort()
    best_new_score, best_candidate = scores_with_candidates[0]
    if current_score > best_new_score:
        remaining_features.remove(best_candidate)
        selected_features_aic.append(best_candidate)
        current_score = best_new_score
    else:
        break
print("Selected features using Forward Selection with AIC:")
print(selected_features_aic)

# Forward Selection with BIC
remaining_features = list(X_train.columns)
selected_features_bic = []
current_score, best_new_score = float('inf'), float('inf')
while remaining_features:
    scores_with_candidates = []
    for candidate in remaining_features:
        features = selected_features_bic + [candidate]
        X_train_const = sm.add_constant(X_train[features])
        model = sm.OLS(y_train, X_train_const).fit()
        scores_with_candidates.append((model.bic, candidate))
    scores_with_candidates.sort()
    best_new_score, best_candidate = scores_with_candidates[0]
    if current_score > best_new_score:
        remaining_features.remove(best_candidate)
        selected_features_bic.append(best_candidate)
        current_score = best_new_score
    else:
        break
print("Selected features using Forward Selection with BIC:")
print(selected_features_bic)

# Comparison of Models
print("\nComparison of Selected Features:")
print("Backward Elimination with AIC Selected Features:")
print(X_train_const.columns)

print("Backward Elimination with BIC Selected Features:")
print(X_train_const_bic.columns)

print("Forward Selection with AIC Selected Features:")
print(selected_features_aic)

print("Forward Selection with BIC Selected Features:")
print(selected_features_bic)


# (d) Ridge Regression with Bootstrap and Cross-Validation
# Cross-Validation to find optimal alpha
alphas = np.logspace(-6, 6, 13)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
print(f"Optimal alpha using Cross-Validation: {ridge_cv.alpha_}")

# Bootstrap to find optimal alpha
bootstrap_alphas = []
for i in range(100):
    X_resampled, y_resampled = resample(X_train, y_train, random_state=i)
    ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_resampled, y_resampled)
    bootstrap_alphas.append(ridge_cv.alpha_)

plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_alphas, kde=True, label='Bootstrap Alphas', color='blue')
plt.axvline(ridge_cv.alpha_, color='red', linestyle='--', label='Cross-Validation Optimal Alpha')
plt.xlabel('Alpha')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution vs Cross-Validation Optimal Alpha for Ridge Regression')
plt.legend()
plt.savefig(f'plot_output_{random.randint(0, 10000)}.png', format='png', dpi=300)
plt.close()

# (e) Generalized Additive Model (GAM)
# Fit a GAM with different levels of complexity
gam_1 = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5) + s(6) + s(7)).fit(X_train, y_train)
gam_2 = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10) + s(2, n_splines=10) + s(3, n_splines=10) +
                  s(4, n_splines=10) + s(5, n_splines=10) + s(6, n_splines=10) + s(7, n_splines=10)).fit(X_train, y_train)

print(f"GAM Model 1 AIC: {gam_1.statistics_['AIC']}")
print(f"GAM Model 2 AIC: {gam_2.statistics_['AIC']}")

# Test set performance
y_test_pred_1 = gam_1.predict(X_test)
y_test_pred_2 = gam_2.predict(X_test)

mse_1 = mean_squared_error(y_test, y_test_pred_1)
mse_2 = mean_squared_error(y_test, y_test_pred_2)

print(f"GAM Model 1 Test MSE: {mse_1}")
print(f"GAM Model 2 Test MSE: {mse_2}")

# (f) Regression Tree with Cost-Complexity Pruning
from sklearn.model_selection import train_test_split

# Fit initial regression tree
reg_tree = DecisionTreeRegressor(random_state=42)
reg_tree.fit(X_train, y_train)

# Split training data into training and validation subsets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Cost-Complexity Pruning Path
path = reg_tree.cost_complexity_pruning_path(X_train_sub, y_train_sub)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train multiple trees with different alpha values
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train_sub, y_train_sub)
    trees.append(tree)

# Evaluate the validation error for each tree
validation_errors = [mean_squared_error(y_val, tree.predict(X_val)) for tree in trees]

# Select the best tree based on validation error
best_tree_idx = np.argmin(validation_errors)
best_tree = trees[best_tree_idx]

print(f"Best tree chosen with ccp_alpha={ccp_alphas[best_tree_idx]}")

# Plot validation error vs complexity parameter alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, validation_errors, marker='o', linestyle='--')
plt.xlabel('Alpha (ccp_alpha)')
plt.ylabel('Validation Error (MSE)')
plt.title('Validation Error vs. Complexity Parameter (Alpha)')
plt.savefig(f'validation_error_vs_alpha_{random.randint(0, 10000)}.png', format='png', dpi=300)
plt.close()

# Plot the best tree
plt.figure(figsize=(20, 10), dpi=300)
plot_tree(best_tree, filled=True, feature_names=X_train.columns, precision=2)
plt.savefig(f'regression_tree_plot_{random.randint(0, 10000)}.svg', format='svg', dpi=300)
plt.close()


# (g) Compare all models in terms of training and test error

# Include all models: Linear Regression, Ridge Regression, Regression Tree, and GAMs
models = {
    "Linear Regression (Direct)": lr_model_1,
    "Linear Regression (One-Hot)": lr_model_2,
    "Ridge Regression": ridge_cv,
    "Regression Tree": best_tree,
    "GAM (Default Splines)": gam_1,
    "GAM (10 Splines)": gam_2,
}

# Create dictionaries to store training and test errors for each model
train_errors = {}
test_errors = {}

for model_name, model in models.items():
    if model_name == "Linear Regression (One-Hot)":
        train_pred = model.predict(X_train_combined)
        test_pred = model.predict(X_test_combined)
    else:
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    train_errors[model_name] = train_error
    test_errors[model_name] = test_error
    print(f"{model_name} - Training Error: {train_error}, Test Error: {test_error}")

# Plotting training and test errors for comparison
plt.figure(figsize=(14, 8))
bar_width = 0.35
index = np.arange(len(models))

train_bar = plt.bar(index, list(train_errors.values()), bar_width, label='Training Error', color='b')
test_bar = plt.bar(index + bar_width, list(test_errors.values()), bar_width, label='Test Error', color='r')

plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors for Different Models')
plt.xticks(index + bar_width / 2, list(models.keys()), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(f'model_comparison_plot_{random.randint(0, 10000)}.png', format='png', dpi=300)
plt.close()

# Load dataset for classification (Problem 2)
pima_data = pd.read_csv('./assignment_2/pimaindiansdiabetes2.csv')

# Drop columns 'triceps' and 'insulin' and remove NaN values
pima_data = pima_data.drop(columns=['triceps', 'insulin']).dropna()

# Split dataset into training and test sets
train_data_pima, test_data_pima = train_test_split(pima_data, test_size=0.33, random_state=42)

# Define features and target for classification
X_train_pima = train_data_pima.drop(columns=['diabetes'])
label_encoder = LabelEncoder()
y_train_pima = label_encoder.fit_transform(train_data_pima['diabetes'])
X_test_pima = test_data_pima.drop(columns=['diabetes'])
y_test_pima = label_encoder.transform(test_data_pima['diabetes'])

# (a) Fit a k-NN classifier
k_values = range(1, 21)
k_scores_5_fold = []

# 5-fold Cross-Validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score_5_fold = cross_val_score(knn, X_train_pima, y_train_pima, cv=5, scoring='accuracy').mean()
    k_scores_5_fold.append(score_5_fold)

# Leave-One-Out Cross-Validation (LOOCV)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
k_scores_loocv = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score_loocv = cross_val_score(knn, X_train_pima, y_train_pima, cv=loo, scoring='accuracy').mean()
    k_scores_loocv.append(score_loocv)

# Calculate Test Errors for different k values
test_errors = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pima, y_train_pima)
    y_test_pred = knn.predict(X_test_pima)
    test_error = 1 - accuracy_score(y_test_pima, y_test_pred)  # Calculate test error as 1 - accuracy
    test_errors.append(test_error)

# Plot cross-validation and test errors
plt.figure(figsize=(10, 6))
plt.plot(k_values, 1 - np.array(k_scores_5_fold), label='5-Fold CV Error', linestyle='--', marker='o')
plt.plot(k_values, 1 - np.array(k_scores_loocv), label='LOOCV Error', linestyle='--', marker='o')
plt.plot(k_values, test_errors, label='Test Error', linestyle='-', marker='s', color='red')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Error Rate')
plt.title('k-NN Classifier - Cross-Validation and Test Errors')
plt.legend()
plt.savefig(f'knn_crossvalidation_test_error_plot_{random.randint(0, 10000)}.png', format='png', dpi=300)
plt.close()

# Select best k and fit k-NN on test data
best_k = k_values[np.argmax(k_scores_5_fold)]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_pima, y_train_pima)
y_test_pred_knn = knn_best.predict(X_test_pima)
test_accuracy_knn = accuracy_score(y_test_pima, y_test_pred_knn)
print(f"Best k: {best_k}, Test Accuracy: {test_accuracy_knn}")

# (b) Fit a GAM with splines and use variable selection
# Fit Logistic GAM
gam_pima = LogisticGAM(s(0) + s(1) + s(2) + s(3) + s(4) + s(5)).fit(X_train_pima, y_train_pima)
print("Selected variables for GAM:")
gam_summary = gam_pima.summary()
print(gam_summary)

# (c) Fit a classification tree, bagged trees, and random forest

# Classification Tree
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train_pima, y_train_pima)
y_train_pred_tree = clf_tree.predict(X_train_pima)
y_test_pred_tree = clf_tree.predict(X_test_pima)
train_accuracy_tree = accuracy_score(y_train_pima, y_train_pred_tree)
test_accuracy_tree = accuracy_score(y_test_pima, y_test_pred_tree)
print(f"Classification Tree - Training Accuracy: {train_accuracy_tree}")
print(f"Classification Tree - Test Accuracy: {test_accuracy_tree}")

# Bagged Trees
bagged_trees = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagged_trees.fit(X_train_pima, y_train_pima)
y_train_pred_bagged = bagged_trees.predict(X_train_pima)
y_test_pred_bagged = bagged_trees.predict(X_test_pima)
train_accuracy_bagged = accuracy_score(y_train_pima, y_train_pred_bagged)
test_accuracy_bagged = accuracy_score(y_test_pima, y_test_pred_bagged)
print(f"Bagged Trees - Training Accuracy: {train_accuracy_bagged}")
print(f"Bagged Trees - Test Accuracy: {test_accuracy_bagged}")

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_pima, y_train_pima)
y_train_pred_rf = random_forest.predict(X_train_pima)
y_test_pred_rf = random_forest.predict(X_test_pima)
train_accuracy_rf = accuracy_score(y_train_pima, y_train_pred_rf)
test_accuracy_rf = accuracy_score(y_test_pima, y_test_pred_rf)
print(f"Random Forest - Training Accuracy: {train_accuracy_rf}")
print(f"Random Forest - Test Accuracy: {test_accuracy_rf}")


# (d) Fit a neural network
from sklearn.model_selection import GridSearchCV

# Define parameter grid for hidden layer sizes
parameter_space = {
    'hidden_layer_sizes': [(10,), (10, 10), (50,), (50, 50), (100,), (100, 50)],
    'max_iter': [1000],
    'random_state': [42]
}

# Set up the MLPClassifier and GridSearchCV
mlp = MLPClassifier()
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, scoring='accuracy')

# Fit the grid search to find the best parameters
clf.fit(X_train_pima, y_train_pima)

# Output the best parameters found
print(f"Best parameters found: {clf.best_params_}")

# Train final model using best parameters
best_mlp = clf.best_estimator_
y_train_pred_best_mlp = best_mlp.predict(X_train_pima)
y_test_pred_best_mlp = best_mlp.predict(X_test_pima)
train_accuracy_best_mlp = accuracy_score(y_train_pima, y_train_pred_best_mlp)
test_accuracy_best_mlp = accuracy_score(y_test_pima, y_test_pred_best_mlp)
print(f"Best Neural Network - Training Accuracy: {train_accuracy_best_mlp}")
print(f"Best Neural Network - Test Accuracy: {test_accuracy_best_mlp}")


# (e) Recommendation of the best model for analysis
print("Based on the test accuracy, the best model will be determined after evaluating the test results of all models.")

import pandas as pd
from ucimlrepo import fetch_ucirepo 

# Fetch dataset 
individual_household_electric_power_consumption = fetch_ucirepo(id=235) 

# Data (as pandas dataframes) 
X = individual_household_electric_power_consumption.data.features 
y = individual_household_electric_power_consumption.data.targets 

# Variable information 
print(individual_household_electric_power_consumption.variables) 

# Combine features and targets for easier manipulation
data = pd.concat([X, y], axis=1)
#NOTE: The above code is taken from UCI's "import in python" function. This 

# Convert 'Date' and 'Time' into a single datetime column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Function to categorize time of day
def categorize_time_of_day(hour):
    return int(hour)

# Apply the function to create a new column 'Time_of_Day'
data['Hour'] = data['Datetime'].dt.hour.apply(categorize_time_of_day)

# Extract the month and create a new column 'Month'
data['Month'] = data['Datetime'].dt.month

# Drop the original Date, Time, and Datetime columns if not needed
data = data.drop(columns=['Date', 'Time', 'Datetime'])

# As the output shows, some of the data is still objects. We therefore need to
# convert it to numerical values.
print(data.dtypes)


cols_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2']

for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#Check for NaN values
print(data.isna().sum())

#Check that all columns now have numerical values (except Time_of_Day column)
print(data.dtypes)

# Drop rows that contain NaN values
data.dropna(axis=0, inplace=True)

##############################


#summary_stats = data.describe(include="all")
#
## Combine the summaries
#pd.set_option('display.max_columns', None)
#summary_stats.loc['count'] = len(data)
#print(summary_stats)


##############################
import matplotlib.pyplot as plt
import numpy as np

# Group data by month and calculate the mean for Global_active_power
monthly_avg = data.groupby('Month')['Global_active_power'].mean()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg.index, monthly_avg.values, color='red', linestyle='--', marker='o', markersize=15)
plt.title('Average Daily Energy Usage', fontsize=20)
plt.xlabel('Month')
plt.ylabel('Energy Usage (kW)')
plt.xticks(ticks=np.arange(1, 13, 1), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()


#import matplotlib.pyplot as plt
#import numpy as np
#
## Group data by month and calculate the mean and standard deviation for Global_active_power
#monthly_avg = data.groupby('Month')['Global_active_power'].mean()
#monthly_std = data.groupby('Month')['Global_active_power'].std()
#
## Plot the data
#plt.figure(figsize=(10, 6))
#plt.errorbar(monthly_avg.index, monthly_avg.values, yerr=monthly_std.values, fmt='-o', color='blue', ecolor='lightgray', elinewidth=2, capsize=4)
#plt.title('Average Monthly Energy Usage with Variability', fontsize=16)
#plt.xlabel('Month', fontsize=14)
#plt.ylabel('Energy Usage (kW)', fontsize=14)
#plt.xticks(ticks=np.arange(1, 13, 1), labels=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
#plt.grid(True, linestyle='--', alpha=0.7)
#plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the features and target variable
X = data[['Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Hour', 'Month']]
y = data['Global_active_power']
possible_features = X.columns

# Split the data into training, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
# Using 50% of the test set as validation set
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=4)

# Feature selection
from itertools import combinations

# Function to generate all combinations of a list
def generate_combinations(lst):
    all_combinations = []
    for r in range(1, len(lst) + 1):
        comb = list(combinations(lst, r))
        all_combinations.extend(comb)
    return all_combinations

# Generate all combinations
all_combinations = generate_combinations(possible_features)

# Convert to list of lists and print the total count
all_combinations_list = [list(comb) for comb in all_combinations]

# Now we will do a training loop where we fit a model for each combination of features, using MSE as selection criteria
best_mse = float('inf')
best_features = None
best_model = None
for i, comb in enumerate(all_combinations_list):
    # Select the features
    X_train_subset = X_train[comb]
    X_val_subset = X_val[comb]
    
    # Fit the model
    model = LinearRegression()
    model.fit(X_train_subset, y_train)
    
    # Predict on the validation set
    y_pred = model.predict(X_val_subset)
    
    # Calculate the mean squared error
    mse = mean_squared_error(y_val, y_pred)
    
    # Update the best model if the current model is better
    if mse < best_mse:
        best_mse = mse
        best_features = comb
        best_model = model
    print(f'Iteration {i+1}/{len(all_combinations_list)} - MSE: {mse}')

print(f'Best features: {best_features}')
print(f'Best MSE: {best_mse}')

# Now it is time to evaluate the best model on the test set to see how good our model really is.
X_test_subset = X_test[best_features]
y_pred = model.predict(X_test_subset)
mse_test = mean_squared_error(y_test, y_pred)
print(f'MSE on test set: {mse_test}')

# Plot the predicted values against the true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()


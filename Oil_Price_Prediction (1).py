#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# In[3]:


import joblib
import pickle


# In[4]:


data=pd.read_csv("combined_df.csv")
data


# In[5]:


column_to_drop = 0
data.drop(data.columns[column_to_drop], axis=1, inplace=True)


# In[6]:


# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])


# In[7]:


data.set_index('Date', inplace=True)


# In[8]:


data.index.year


# In[9]:


data


# In[10]:


plt.figure(figsize=(15,5))
sns.lineplot(data)


# In[11]:


# Checking Missing Values
data.isnull().sum()


# In[12]:


# Handling Missing Values
data['Price'] = data['Price'].interpolate()


# In[13]:


data


# In[14]:


data.isnull().sum()


# In[15]:


# Checking for Negative Value
(data['Price'] < 0).sum()


# In[16]:


# Removing Negative Values
data['Price'] = data['Price'].apply(lambda x: x if x >= 0 else None)


# In[17]:


# Outliers
sns.boxplot(data)
plt.show()


# In[18]:


Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
outliers = data[(data['Price'] < Q1 - 1.5 * IQR) | (data['Price'] > Q3 + 1.5 * IQR)]


# In[19]:


# Outliers
outliers


# In[20]:


data.dropna(inplace = True)


# #  Stationarity

# In[21]:


# Function to check stationarity and perform differencing
def check_stationarity(data):
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Original Time Series')
    plt.title('Original Time Series')
    plt.show()


# In[22]:


# Perform Dickey-Fuller test
result = adfuller(data, autolag='AIC')
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])


# In[23]:


# Check for stationarity
if result[1] <= 0.05:
    print('The time series is stationary.')
else:
    print('The time series is not stationary.')


# In[24]:


# First-order differencing
data_diff = data.diff().dropna()


# In[25]:


# Plot differenced time series
plt.figure(figsize=(18, 5))
plt.plot(data_diff, label='Differenced Time Series')
plt.title('First-order Differenced Time Series')
plt.show()


# In[26]:


# Perform Dickey-Fuller test on differenced time series
result_diff = adfuller(data_diff, autolag='AIC')
print('ADF Statistic (after differencing):', result_diff[0])
print('p-value (after differencing):', result_diff[1])
print('Critical Values (after differencing):', result_diff[4])


# In[27]:


# Check for stationarity after differencing
if result_diff[1] <= 0.05:
    print('The differenced time series is stationary.')
else:
    print('Differencing did not make the time series stationary.')


# In[28]:


time_series = data['Price']

check_stationarity(time_series)


# # KNN

# In[29]:


data_values = data['Price'].values


# In[30]:


# Choosing the number of neighbors (k) and the lookback window size
k = 5
lookback_window = 10


# In[31]:


# Prepare the feature matrix X and target variable y
X, y = [], []


# In[32]:


for i in range(len(data_values) - lookback_window):
    X.append(data_values[i:i+lookback_window])
    y.append(data_values[i+lookback_window])


# In[33]:


X, y = np.array(X), np.array(y)


# In[34]:


# Scale the data (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[35]:


# Initialize the KNN model
knn_model = KNeighborsRegressor(n_neighbors=k)


# In[36]:


# Train the model
knn_model.fit(X_scaled, y)


# In[37]:


# Assuming you have an array 'actual_values' with the actual values for the previous and next 365 days

forecasted_values = []
actual_values = []  # Replace this with your actual data

# Assuming 'actual_values' already contains data for the previous 365 days
actual_values.extend(data_values[-365:])

for _ in range(365):
    # Use the last 'lookback_window' values as input for the next prediction
    last_window = data_values[-lookback_window:]
    last_window_scaled = scaler.transform(last_window.reshape(1, -1))
    
    # Make a one-step prediction
    next_pred = knn_model.predict(last_window_scaled)
    
    # Append the prediction to the forecasted values
    forecasted_values.append(next_pred[0])
    
    # Update the time series with the new prediction
    data_values = np.append(data_values, next_pred)
    
    # Assuming you have actual values for the next 365 days
    actual_value = actual_values.pop(0)  # Get the actual value for the current day
    actual_values.append(actual_value)  # Move to the next day


# In[38]:


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
print(f'RMSE: {rmse}')


# In[43]:


# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({'Date':pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=365),'Price':forecasted_values})


# In[44]:


# Concatenate the forecast DataFrame with the original dataset
merged_data = pd.concat([data, forecast_df], axis=0)


# In[45]:


merged_data.tail(365)


# In[46]:


merged_data.to_csv(r"C:\Users\Punith sai\OneDrive\Desktop\merged_data.csv")


# In[47]:


# Visualize the merged data
plt.plot(data['Price'], label='Original Data')
plt.plot(merged_data['Price'].tail(365), label='Forecasted Values', linestyle='dashed', color = 'black')
plt.legend()
plt.show()


# In[ ]:





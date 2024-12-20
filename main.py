# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.colab import drive

file_path = '/content/Rotten_Tomatoes_Movies3.csv'
df = pd.read_csv(file_path)

# Exploring the dataset
# Check the first 5 rows of the data set
print("_____First 5 rows______")
print(df.head())

# Check the coloumn names
print("\n _____Coloum names_____ ")
print(df.columns)

print("\n _______dataset information______ ")
print(df.info())
print("\n ______Summary_______")
print(df.describe())
print("\n ________Look for unique values in audience_rating column________")
print(df['audience_rating'].unique())

print("\n _____if missing values______ ")
print(df.isnull().sum)

# We can do a quick visualization of the target variable
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
sns.histplot(df['audience_rating'],bins=20,kde=True)
plt.title('Distribution of Audience Rating')
plt.xlabel('Audience Rating')
plt.ylabel('Frequency')
plt.show()

# Inspect unique values if we find soemthing
cat_cols = df.select_dtypes(include=['object']).columns
print("\n ______Unique values in categorical columns_______")
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# Now we can move ahead with the data cleaning process
print("\n __________Check if we got null values____________")
print(df.isnull().sum())

# we can drop the coloumn which has excessive missing values
df = df.drop(columns=['critics_consensus', 'writers'], axis=1)

# Now check for the remaining values
print("\n _______Remaning Values_______")
print(df.columns)

# Handle missing values in remaining columns
df = df.dropna(subset=['audience_rating'])
df = df.fillna(method='ffill')

# Correlation Heatmap
numerical_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Target Variable
sns.histplot(df['audience_rating'], bins=20, kde=True)
plt.title('Distribution of Audience Rating')
plt.show()

# Pair plot for selected numerical features (modify features based on your dataset)
selected_features = ['tomatometer_rating', 'tomatometer_count', 'audience_rating']
sns.pairplot(df[selected_features])
plt.show()

from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Encode Categorical Features
encoder = LabelEncoder()
categorical_columns = ['rating', 'genre']  # Add relevant columns

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))  # Ensure strings for encoding

# 2. Scale Numerical Features
scaler = StandardScaler()
numerical_columns = ['runtime_in_minutes', 'tomatometer_rating', 'tomatometer_count']

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# View final dataset
print(df.head())

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df.drop(columns=['audience_rating', 'movie_title', 'movie_info', 'directors', 'cast', 'in_theaters_date', 'on_streaming_date', 'studio_name'])
y = df['audience_rating']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['tomatometer_status'] = encoder.fit_transform(df['tomatometer_status'])

print(X_train.dtypes)
print(X_train.head())

from sklearn.model_selection import train_test_split

# Redefine X and y after encoding
X = df.drop(columns=['audience_rating', 'movie_title', 'movie_info', 'directors', 'cast',
                     'in_theaters_date', 'on_streaming_date', 'studio_name'])  # Features
y = df['audience_rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify that X_train is numeric
print(X_train.dtypes)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
print("Linear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

from sklearn.preprocessing import LabelEncoder

# Encode 'tomatometer_status'
encoder = LabelEncoder()
df['tomatometer_status'] = encoder.fit_transform(df['tomatometer_status'])

# Drop unnecessary columns and redefine X and y
X = df.drop(columns=['audience_rating', 'movie_title', 'movie_info', 'directors', 'cast',
                     'in_theaters_date', 'on_streaming_date', 'studio_name'])  # Features
y = df['audience_rating']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify that X_train is numeric
print(X_train.dtypes)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_rf_pred = rf.predict(X_test)

# Evaluate the model
print("Random Forest Performance:")
print("MAE:", mean_absolute_error(y_test, y_rf_pred))
print("MSE:", mean_squared_error(y_test, y_rf_pred))
print("R² Score:", r2_score(y_test, y_rf_pred))

print(X_train.head())
print(X_train.dtypes)

# Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

from sklearn.pipeline import Pipeline

# Full pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=200, max_depth=10))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict and Evaluate
y_pipe_pred = pipeline.predict(X_test)
print("Pipeline R2 Score:", r2_score(y_test, y_pipe_pred))
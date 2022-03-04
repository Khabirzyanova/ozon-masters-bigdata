from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV

import tensorflow as tf
from keras.utils.np_utils import to_categorical


# Dataset fields
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

fields = ["id", "label"] + numeric_features + categorical_features



# Model pipeline
# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Imputation transformer for completing missing values.
# If “median”, then replace missing values using the median along each column. Can only be used with numeric data.
#    ('scaler', StandardScaler()) # Standardize features by removing the mean and scaling to unit variance.
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # If “constant”, then replace missing values with fill_value. 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Encode categorical features as a one-hot numeric array.
])

# Applies transformers to columns of an array or pandas DataFrame.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LinearRegression())
])

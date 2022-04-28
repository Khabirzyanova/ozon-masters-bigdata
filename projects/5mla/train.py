import os, sys
import pandas as pd
import logging

from sklearn.model_selection import train_test_split
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import log_loss

import mlflow
import argparse


def parse_args():
   parser = argparse.ArgumentParser(description="LogRegr model")
   parser.add_argument("--train_path", type=str)
   parser.add_argument("--model_param1", type=float, default=1.0)
   return parser.parse_args() 

def main():
   args = parse_args()
   train_path = args.train_path 
   C = args.model_param1
   numeric_features = ["if"+str(i) for i in range(1,14)]
   categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
   fields = ["id", "label"] + numeric_features + categorical_features
   fields_val = ["id"] + numeric_features + categorical_features
   
   read_table_opts = dict(sep="\t", names=fields, index_col=False)
   df = pd.read_table(train_path, **read_table_opts)
   labels = df['label']
   X_train, X_test, y_train, y_test = train_test_split(df.iloc[:1000, 2:], 
                                                       labels.iloc[:1000], 
                                                       test_size=0.33, 
                                                       random_state=42)

   with mlflow.start_run():

      numeric_transformer = Pipeline(steps=[
                                            ('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())
                                           ])

      categorical_transformer = Pipeline(steps=[
                                                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                               ])      

      preprocessor = ColumnTransformer(transformers=[
                                                     ('num', numeric_transformer, numeric_features),
                                                     ('cat', categorical_transformer, categorical_features)
                                                    ]
                                       )

      model = Pipeline(steps=[
                              ('preprocessor', preprocessor),
                              ('logisticregression', LogisticRegression(C = C))
                             ])

      mlflow.log_param("model_param1", args.model_param1)


      model.fit(X_train, y_train)

      mlflow.sklearn.log_model(model, artifact_path="model")
      y_pred = model.predict(X_test)
      score = log_loss(y_test, y_pred)

      mlflow.log_metrics({"log_loss": score})


if __name__ == "__main__":
   main()

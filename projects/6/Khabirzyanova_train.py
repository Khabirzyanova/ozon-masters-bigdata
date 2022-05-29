#!/opt/conda/envs/dsenv/bin/python

from joblib import dump
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os, sys

df_path = sys.argv[2]
path_to_save = sys.argv[4]



fields = ["id", "label", "intercept", "unixReviewTime"]

model = LogisticRegression()



df = pd.read_parquet(df_path, columns=fields)



# data[['label']] = df[['label']]

# NANs
df.fillna(0, inplace=True)

y = df.label
X = df[["intercept", "unixReviewTime"]]

model.fit(X, y)

# save the model
dump(model, path_to_save)

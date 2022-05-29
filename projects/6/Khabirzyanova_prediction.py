#!/opt/conda/envs/dsenv/bin/python

import pandas as pd
import sys, os


SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

# start session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')


test_in  = sys.argv[2]
pred_out = sys.argv[4]
sklearn_model_in = sys.argv[6]

from joblib import load
#load the model
model = load(sklearn_model_in)



from pyspark.sql.types import *

log_schema = StructType(fields=[
    StructField("id", StringType()),
    StructField("unixReviewTime", FloatType())])

df = spark.read.parquet(test_in, schema=log_schema)

from pyspark.sql.functions import udf
from pyspark.sql.functions import PandasUDFType

@udf(returnType=IntegerType())
def predict_udf(*columns):
    return int(model.predict((columns, )))


df = df.withColumn("prediction", predict_udf("intercept", "unixReviewTime"))
df.select('id', 'prediction').write.mode("overwrite").save(pred_out, header='false', format='csv')

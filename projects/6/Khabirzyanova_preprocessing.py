import os
import sys

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



# create dataframe
from pyspark.sql.types import *

log_schema = StructType(fields=[
    StructField("id", StringType()),
    StructField("label", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", FloatType())])

file_name = sys.argv[2]
df = spark.read.json(file_name, schema=log_schema)

from pyspark.sql.functions import lit

df2 = df.withColumn("intercept", lit(1))

df2.select("id", "label", "intercept", "unixReviewTime").write.mode("overwrite").parquet(sys.argv[4])

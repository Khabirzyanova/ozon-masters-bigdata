import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, \
                              ArrayType, LongType, \
                              IntegerType  
import pyspark.sql.functions as F

spark = SparkSession.builder.config(conf=SparkConf()).\
                     appName("BFS").getOrCreate()

context = spark.sparkContext

schema = StructType(fields=[
    StructField("vertex_to", IntegerType()),
    StructField("vertex_from", IntegerType())
])

df = spark.read.schema(schema).format("csv").\
           option("sep", "\t").load(sys.argv[3])

v_from = sys.argv[1]
v_to = sys.argv[2]

max_path_length = df.select("vertex_to").distinct().count()

queue = df.where(F.col("vertex_from") == v_from).\
           withColumnRenamed("vertex_from", "vertex_end_0")\
           .withColumnRenamed("vertex_to", "vertex_from")

sc = spark.sparkContext

df_end = spark.createDataFrame(
        sc.emptyRDD(),
        StructType([StructField("path", ArrayType(LongType()), True)])
    )

for i in range(1, max_path_length + 1):
   queue = queue.join(df, "vertex_from")\
                    .withColumnRenamed("vertex_from", "vertex_end_" + str(i))\
                    .withColumnRenamed("vertex_to", "vertex_from")
   cur_df = queue.where(F.col("vertex_from") == v_to)
   indexes = []
   for column in queue.columns:
      if column != "vertex_from":
         cur_ind = int(column.split('_')[2])
         indexes.append(cur_ind)

   indexes = sorted(indexes)
   sorted_columns = ["vertex_end_" + str(i) for i in indexes] + ["vertex_from"]
   df_end = cur_df.select(F.array(*(sorted_columns)).\
                   alias("path")).unionAll(df_end)
   if df_end.select("*").count():
      break

df_end = df_end.withColumn('path', F.concat_ws(',', 'path'))
df_end.select("path").write.mode("overwrite").text(sys.argv[4])
spark.stop()

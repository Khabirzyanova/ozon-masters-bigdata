#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, VectorAssembler, StringIndexer, StopWordsRemover, CountVectorizer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

stop_words = StopWordsRemover.loadDefaultStopWords("english")

tokenizer = Tokenizer(inputCol="reviewText", outputCol="reviewWords")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="reviewFiltered", stopWords=stop_words)
vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="reviewVec", minDF=0.005)

indexer1 = StringIndexer(inputCol="asin", outputCol="asinLabeled")

assembler = VectorAssembler(
    inputCols=['reviewVec', "asinLabeled"],
    outputCol="features")

lr = LinearRegression(maxIter=200, regParam=0.01, elasticNetParam=0.1, labelCol="overall")

pipeline = Pipeline(stages=[tokenizer, swr, vectorizer, indexer1, assembler, lr])

# coding=utf-8

from pyspark_hnsw.linalg import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext

def test_normalizer(spark_context):

    sql_context = SQLContext(spark_context)

    df = sql_context.createDataFrame([[Vectors.dense([0.01, 0.02, 0.03])]], ['vector'])

    normalizer = Normalizer(inputCol="vector", outputCol="normalized_vector")

    result = normalizer.transform(df)

    assert result.count() == 1

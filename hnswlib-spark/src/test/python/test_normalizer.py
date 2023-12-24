# coding=utf-8

from pyspark_hnsw.linalg import Normalizer
from pyspark.ml.linalg import Vectors

def test_normalizer(spark):

    df = spark.createDataFrame([[Vectors.dense([0.01, 0.02, 0.03])]], ['vector'])

    normalizer = Normalizer(inputCol="vector", outputCol="normalized_vector")

    result = normalizer.transform(df)

    assert result.count() == 1

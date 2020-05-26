# coding=utf-8

from pyspark_hnsw.knn import HnswSimilarity
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext

from tempfile import gettempdir


def test_hnsw(spark_context):

    spark_context.setCheckpointDir(gettempdir())

    sql_context = SQLContext(spark_context)

    df = sql_context.createDataFrame([
        [1, Vectors.dense([0.2, 0.9])],
        [2, Vectors.dense([0.2, 1.0])],
        [3, Vectors.dense([0.2, 0.1])],
    ], ['row_id', 'features'])

    hnsw = HnswSimilarity(identifierCol='row_id', featuresCol='features', distanceFunction='cosine', m=32, ef=5, k=5,
                          efConstruction=200, numPartitions=100, excludeSelf=False, similarityThreshold=-1.0)

    model = hnsw.fit(df)

    result = model.transform(df)

    assert result.count() == 3

# coding=utf-8

from pyspark_hnsw.knn import HnswSimilarity
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F


def test_incremental_models(spark, tmp_path):

    df1 = spark.createDataFrame([
        [1, Vectors.dense([0.1, 0.2, 0.3])]
    ], ['id', 'features'])

    hnsw1 = HnswSimilarity()

    model1 = hnsw1.fit(df1)

    model1.write().overwrite().save(tmp_path.as_posix())

    df2 = spark.createDataFrame([
        [2, Vectors.dense([0.9, 0.1, 0.2])]
    ], ['id', 'features'])

    hnsw2 = HnswSimilarity(initialModelPath=tmp_path.as_posix())

    model2 = hnsw2.fit(df2)

    assert model2.transform(df1).select(F.explode("prediction")).count() == 2


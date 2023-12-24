# coding=utf-8

from pyspark_hnsw.knn import BruteForceSimilarity
from pyspark.ml.linalg import Vectors


def test_bruteforce(spark):

    df = spark.createDataFrame([
        [1, Vectors.dense([0.2, 0.9])],
        [2, Vectors.dense([0.2, 1.0])],
        [3, Vectors.dense([0.2, 0.1])],
    ], ['row_id', 'features'])

    bruteforce = BruteForceSimilarity(identifierCol='row_id', queryIdentifierCol='row_id', featuresCol='features',
                                      distanceFunction='cosine', numPartitions=100, excludeSelf=False,
                                      similarityThreshold=-1.0)

    model = bruteforce.fit(df)

    result = model.transform(df)

    assert result.count() == 3

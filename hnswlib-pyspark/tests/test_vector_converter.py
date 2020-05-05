# coding=utf-8

from pyspark_hnsw.conversion import VectorConverter
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext

def test_vector_converter(spark_context):

    sql_context = SQLContext(spark_context)

    df = sql_context.createDataFrame([[Vectors.dense([0.01, 0.02, 0.03])]], ['vector'])

    converter = VectorConverter(inputCol="vector", outputCol="array")

    result = converter.transform(df)

    assert result.count() == 1

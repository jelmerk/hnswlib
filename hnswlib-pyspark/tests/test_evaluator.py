# coding=utf-8

from pyspark_hnsw.evaluator import KnnEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
from pyspark.sql.types import *


def test_evaluator(spark_context):

    sql_context = SQLContext(spark_context)

    neighbors_list_schema = ArrayType(StructType([StructField("neighbor", IntegerType()), StructField("distance", FloatType())]))

    schema = StructType([StructField("approximate", neighbors_list_schema), StructField("exact", neighbors_list_schema)])

    df = sql_context.createDataFrame([
        [[{'neighbor': 1, 'distance': 0.1}], [{'neighbor': 1, 'distance': 0.1}]],
        [[{'neighbor': 2, 'distance': 0.1}], [{'neighbor': 2, 'distance': 0.1}, {'neighbor': 3, 'distance': 0.9}]]
    ], schema=schema)

    evaluator = KnnEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')

    assert evaluator.evaluate(df) == 0.75


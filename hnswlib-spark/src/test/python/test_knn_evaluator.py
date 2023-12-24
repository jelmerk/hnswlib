# coding=utf-8

from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark.sql.types import *


def test_evaluator(spark):

    neighbors_list_schema = ArrayType(StructType([StructField("neighbor", IntegerType()), StructField("distance", FloatType())]))

    schema = StructType([StructField("approximate", neighbors_list_schema), StructField("exact", neighbors_list_schema)])

    df = spark.createDataFrame([
        [[{'neighbor': 1, 'distance': 0.1}], [{'neighbor': 1, 'distance': 0.1}]],
        [[{'neighbor': 2, 'distance': 0.1}], [{'neighbor': 2, 'distance': 0.1}, {'neighbor': 3, 'distance': 0.9}]]
    ], schema=schema)

    evaluator = KnnSimilarityEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')

    assert evaluator.evaluate(df) == 0.6666666666666666


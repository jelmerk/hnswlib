# -*- coding: utf-8 -*-

import argparse

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark_hnsw.knn import *
from pyspark_hnsw.linalg import Normalizer


def main(spark):
    parser = argparse.ArgumentParser(description='Construct hnsw index')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--m', type=int)
    parser.add_argument('--ef_construction', type=int)
    parser.add_argument('--num_partitions', type=int)

    args = parser.parse_args()

    normalizer = Normalizer(inputCol='features', outputCol='normalized_features')

    hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features',
                          distanceFunction='inner-product', m=args.m, efConstruction=args.ef_construction,
                          numPartitions=args.num_partitions, excludeSelf=True, outputFormat='minimal')

    pipeline = Pipeline(stages=[normalizer, hnsw])

    index_items = spark.read.parquet(args.input)

    model = pipeline.fit(index_items)

    model.write().overwrite().save(args.output)


if __name__ == '__main__':
    main(SparkSession.builder.getOrCreate())

# -*- coding: utf-8 -*-

import argparse

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark_hnsw.knn import *
from pyspark_hnsw.linalg import Normalizer


def main(spark):
    parser = argparse.ArgumentParser(description='Construct brute force index')
    parser.add_argument('--input', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--num_partitions', type=int)

    args = parser.parse_args()

    normalizer = Normalizer(inputCol='features', outputCol='normalized_features')

    bruteforce = BruteForceSimilarity(identifierCol='id', featuresCol='normalized_features',
                                      distanceFunction='inner-product', numPartitions=args.num_partitions)

    pipeline = Pipeline(stages=[normalizer, bruteforce])

    index_items = spark.read.parquet(args.input)

    model = pipeline.fit(index_items)

    model.write().overwrite().save(args.output)


if __name__ == '__main__':
    main(SparkSession.builder.getOrCreate())

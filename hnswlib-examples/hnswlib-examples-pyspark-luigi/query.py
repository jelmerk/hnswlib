# -*- coding: utf-8 -*-

import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


def main(spark):
    parser = argparse.ArgumentParser(description='Query index')
    parser.add_argument('--input', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--ef', type=int)
    parser.add_argument('--num_replicas', type=int)

    args = parser.parse_args()

    model = PipelineModel.read().load(args.model)

    hnsw_stage = model.stages[-1]
    hnsw_stage.setEf(args.ef)
    hnsw_stage.setK(args.k)
    hnsw_stage.setNumReplicas(args.num_replicas)

    query_items = spark.read.parquet(args.input)

    results = model.transform(query_items)

    results.write.mode('overwrite').json(args.output)


if __name__ == '__main__':
    main(SparkSession.builder.getOrCreate())

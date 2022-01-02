# -*- coding: utf-8 -*-

import argparse

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator


def main(spark):
    parser = argparse.ArgumentParser(description='Evaluate performance of the index')
    parser.add_argument('--hnsw_model', type=str)
    parser.add_argument('--bruteforce_model', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--ef', type=int)
    parser.add_argument('--fraction', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    sample_query_items = spark.read.parquet(args.input).sample(False, args.fraction, args.seed)

    hnsw_model = PipelineModel.read().load(args.hnsw_model)

    hnsw_stage = hnsw_model.stages[-1]
    hnsw_stage.setEf(args.ef)
    hnsw_stage.setK(args.k)
    hnsw_stage.setPredictionCol('approximate')
    hnsw_stage.setOutputFormat('full')

    bruteforce_model = PipelineModel.read().load(args.bruteforce_model)

    bruteforce_stage = bruteforce_model.stages[-1]
    bruteforce_stage.setK(args.k)
    bruteforce_stage.setPredictionCol('exact')
    bruteforce_stage.setOutputFormat('full')

    sample_results = bruteforce_model.transform(hnsw_model.transform(sample_query_items))

    evaluator = KnnSimilarityEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')

    accuracy = evaluator.evaluate(sample_results)

    spark.createDataFrame([[accuracy]], ['accuracy']).repartition(1).write.mode('overwrite').csv(args.output)


if __name__ == '__main__':
    main(SparkSession.builder.getOrCreate())

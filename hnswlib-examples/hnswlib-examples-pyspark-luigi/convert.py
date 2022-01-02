# -*- coding: utf-8 -*-

import argparse

from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark_hnsw.conversion import VectorConverter


def main(spark):
    parser = argparse.ArgumentParser(description='Convert input file to parquet')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    words_df = spark.read \
        .option('inferSchema', 'true') \
        .option('quote', '\u0000') \
        .option('delimiter', ' ') \
        .csv(args.input) \
        .withColumnRenamed('_c0', 'id')

    vector_assembler = VectorAssembler(inputCols=words_df.columns[1:], outputCol='features_as_vector')

    converter = VectorConverter(inputCol='features_as_vector', outputCol='features', outputType='array<float>')

    converter.transform(vector_assembler.transform(words_df)) \
        .select('id', 'features') \
        .write \
        .parquet(args.output)


if __name__ == "__main__":
    main(SparkSession.builder.getOrCreate())

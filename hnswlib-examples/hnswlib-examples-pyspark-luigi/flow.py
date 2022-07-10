# -*- coding: utf-8 -*-

import urllib.request
import shutil

import luigi
from luigi import FloatParameter, IntParameter, LocalTarget, Parameter
from luigi.contrib.spark import SparkSubmitTask
from luigi.format import Nop
from luigi.contrib.external_program import ExternalProgramTask
# from luigi.contrib.hdfs import HdfsFlagTarget
# from luigi.contrib.s3 import S3FlagTarget


class Download(luigi.Task):
    """
    Download the input dataset.
    """

    url = Parameter(default='https://nlp.stanford.edu/data/glove.42B.300d.zip')

    def output(self):
        return LocalTarget('/tmp/dataset.zip', format=Nop)

    def run(self):
        # noinspection PyTypeChecker
        with urllib.request.urlopen(self.url) as response:
            with self.output().open('wb') as f:
                shutil.copyfileobj(response, f)


class Unzip(ExternalProgramTask):
    """
    Unzip the input dataset.
    """

    def requires(self):
        return Download()

    def output(self):
        return LocalTarget('/tmp/dataset', format=Nop)

    def program_args(self):
        self.output().makedirs()
        return ['unzip',
                '-u',
                '-q',
                '-d', self.output().path,
                self.input().path]


class Convert(SparkSubmitTask):
    """
    Convert the input dataset to parquet.
    """

    # master = 'yarn'
    master = 'local[*]'

    deploy_mode = 'client'

    driver_memory = '2g'

    # executor_memory = '4g'

    num_executors = IntParameter(default=2)

    name = 'Convert'

    app = 'convert.py'

    packages = ['com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0']

    def requires(self):
        return Unzip()

    def app_options(self):
        return [
            "--input", self.input().path,
            "--output", self.output().path
        ]

    def output(self):
        # return HdfsFlagTarget('/tmp/vectors_parquet')
        # return S3FlagTarget('/tmp/vectors_parquet')
        return LocalTarget('/tmp/vectors_parquet', format=Nop)


class HnswIndex(SparkSubmitTask):
    """
    Construct the hnsw index and persists it to disk.
    """

    # master = 'yarn'
    master = 'local[*]'

    deploy_mode = 'client'

    # driver_memory = '2g'
    driver_memory = '24g'

    # executor_memory = '12g'

    num_executors = IntParameter(default=2)

    executor_cores = IntParameter(default=2)

    name = 'Hnsw index'

    app = 'hnsw_index.py'

    packages = ['com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0']

    m = IntParameter(default=16)

    ef_construction = IntParameter(default=200)

    @property
    def conf(self):
        return {'spark.dynamicAllocation.enabled': 'false',
                'spark.speculation': 'false',
                'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
                'spark.kryo.registrator': 'com.github.jelmerk.spark.HnswLibKryoRegistrator',
                'spark.task.cpus': str(self.executor_cores),
                'spark.task.maxFailures': '1',
                'spark.scheduler.minRegisteredResourcesRatio': '1.0',
                'spark.scheduler.maxRegisteredResourcesWaitingTime': '3600s',
                'spark.hnswlib.settings.index.cache_folder': '/tmp'}

    def requires(self):
        return Convert()

    def app_options(self):
        return [
            '--input', self.input().path,
            '--output', self.output().path,
            '--m', self.m,
            '--ef_construction', self.ef_construction,
            '--num_partitions', str(self.num_executors)
        ]

    def output(self):
        # return HdfsFlagTarget('/tmp/hnsw_index')
        # return S3FlagTarget('/tmp/hnsw_index')
        return LocalTarget('/tmp/hnsw_index', format=Nop)


class Query(SparkSubmitTask):
    """
    Query the constructed knn index.
    """

    # master = 'yarn'
    master = 'local[*]'

    deploy_mode = 'client'

    # driver_memory = '2g'
    driver_memory = '24g'

    # executor_memory = '10g'

    num_executors = IntParameter(default=4)

    executor_cores = IntParameter(default=2)

    packages = ['com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0']

    name = 'Query index'

    app = 'query.py'

    k = IntParameter(default=10)

    ef = IntParameter(default=100)

    num_replicas = IntParameter(default=1)

    @property
    def conf(self):
        return {'spark.dynamicAllocation.enabled': 'false',
                'spark.speculation': 'false',
                'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
                'spark.kryo.registrator': 'com.github.jelmerk.spark.HnswLibKryoRegistrator',
                'spark.task.cpus': str(self.executor_cores),
                'spark.task.maxFailures': '1',
                'spark.scheduler.minRegisteredResourcesRatio': '1.0',
                'spark.scheduler.maxRegisteredResourcesWaitingTime': '3600s'}

    def requires(self):
        return {'vectors': Convert(),
                'index': HnswIndex()}

    def app_options(self):
        return [
            '--input', self.input()['vectors'].path,
            '--model', self.input()['index'].path,
            '--output', self.output().path,
            '--ef', self.ef,
            '--k', self.k,
            '--num_replicas', self.num_replicas
        ]

    def output(self):
        # return HdfsFlagTarget('/tmp/query_results')
        # return S3FlagTarget('/tmp/query_results')
        return LocalTarget('/tmp/query_results')


class BruteForceIndex(SparkSubmitTask):
    """
    Construct the brute force index and persists it to disk.
    """

    # master = 'yarn'
    master = 'local[*]'

    deploy_mode = 'client'

    # driver_memory = '2g'
    driver_memory = '24g'

    # executor_memory = '12g'

    num_executors = IntParameter(default=2)

    executor_cores = IntParameter(default=2)

    name = 'Brute force index'

    app = 'bruteforce_index.py'

    packages = ['com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0']

    @property
    def conf(self):
        return {'spark.dynamicAllocation.enabled': 'false',
                'spark.speculation': 'false',
                'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
                'spark.kryo.registrator': 'com.github.jelmerk.spark.HnswLibKryoRegistrator',
                'spark.task.cpus': str(self.executor_cores),
                'spark.task.maxFailures': '1',
                'spark.scheduler.minRegisteredResourcesRatio': '1.0',
                'spark.scheduler.maxRegisteredResourcesWaitingTime': '3600s',
                'spark.hnswlib.settings.index.cache_folder': '/tmp'}

    def requires(self):
        return Convert()

    def app_options(self):
        return [
            '--input', self.input().path,
            '--output', self.output().path,
            '--num_partitions', str(self.num_executors)
        ]

    def output(self):
        # return HdfsFlagTarget('/tmp/brute_force_index')
        # return S3FlagTarget('/tmp/brute_force_index')
        return LocalTarget('/tmp/brute_force_index', format=Nop)


class Evaluate(SparkSubmitTask):
    """
    Evaluate the accuracy of the approximate k-nearest neighbors model vs a bruteforce baseline.
    """

    # master = 'yarn'
    master = 'local[*]'

    deploy_mode = 'client'

    # driver_memory = '2g'
    driver_memory = '24g'

    # executor_memory = '12g'

    num_executors = IntParameter(default=2)

    executor_cores = IntParameter(default=2)

    k = IntParameter(default=10)

    ef = IntParameter(default=100)

    fraction = FloatParameter(default=0.0001)

    seed = IntParameter(default=123)

    name = 'Evaluate performance'

    app = 'evaluate_performance.py'

    packages = ['com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0']

    @property
    def conf(self):
        return {'spark.dynamicAllocation.enabled': 'false',
                'spark.speculation': 'false',
                'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
                'spark.kryo.registrator': 'com.github.jelmerk.spark.HnswLibKryoRegistrator',
                'spark.task.cpus': str(self.executor_cores),
                'spark.task.maxFailures': '1',
                'spark.scheduler.minRegisteredResourcesRatio': '1.0',
                'spark.scheduler.maxRegisteredResourcesWaitingTime': '3600s'}

    def requires(self):
        return {'vectors': Convert(),
                'hnsw_index': HnswIndex(),
                'bruteforce_index': BruteForceIndex()}

    def app_options(self):
        return [
            '--input', self.input()['vectors'].path,
            '--output', self.output().path,
            '--hnsw_model', self.input()['hnsw_index'].path,
            '--bruteforce_model', self.input()['bruteforce_index'].path,
            '--ef', self.ef,
            '--k', self.k,
            '--seed', self.seed,
            '--fraction', self.fraction,
        ]

    def output(self):
        # return HdfsFlagTarget('/tmp/metrics')
        # return S3FlagTarget('/tmp/metrics')
        return LocalTarget('/tmp/metrics', format=Nop)

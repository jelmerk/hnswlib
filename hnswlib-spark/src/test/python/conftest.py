# coding=utf-8

import os

import pytest

from pyspark.sql import SparkSession

@pytest.fixture(scope="session", autouse=True)
def spark(request):
    sc = SparkSession.builder \
        .config("spark.driver.extraClassPath", os.environ["ARTIFACT_PATH"]) \
        .master("local[*]") \
        .getOrCreate()

    request.addfinalizer(lambda: sc.stop())

    return sc

#!/usr/bin/env bash

set -e

SPARK_VERSION=$1
PYTHON_VERSION=$2

# add python sources on the path
export PYTHONPATH=src/main/python

# unset SPARK_HOME or it will use whatever is configured on the host system instead of the pip packages
unset SPARK_HOME

# create a virtual environment

eval "$PYTHON_VERSION -m venv "target/spark-$SPARK_VERSION-venv""
source "target/spark-$SPARK_VERSION-venv/bin/activate"

# install packages
pip install pytest==7.4.3
pip install 'pyspark[ml]'=="$SPARK_VERSION"

# run unit tests
pytest --junitxml=target/test-reports/TEST-python.xml
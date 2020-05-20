Development
-----------

You can run the integration tests by first running the following command in the root of the project

    mvn clean install

Followed by running the following commands in the hnswlib-pyspark module

    export SPARK_HOME=/path/to/spark-2.3.0-bin-hadoop2.6
    rm -rf ~/.ivy2/cache/com.github.jelmerk
    rm -rf ~/.ivy2/jars/com.github.jelmerk_hnswlib-*
    pip install -e .[test]
    py.test

The easiest way to test changes on a real cluster is to produce an egg file with

    python setup.py bdist_egg
    
And then reference it from spark with

    spark-submit --py-files hnswlib-pyspark/dist/pyspark_hnsw-*.egg YOUR_SCRIPT

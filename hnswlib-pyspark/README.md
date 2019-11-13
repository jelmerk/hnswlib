hnswlib-pyspark
===============

[PySpark](https://spark.apache.org/) integration for hnswlib.

Setup
-----

Pass the following argument to spark

    --packages 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.23'

Then install the python module with

    pip install pyspark-hnsw --upgrade
    

Example usage
-------------

    from pyspark_hnsw.hnsw import Hnsw
    
    hnsw = Hnsw(identifierCol='row_id', vectorCol='features', distanceFunction='cosine', m=32, ef=5, k=5,
                efConstruction=200, numPartitions=100, excludeSelf=False)
    
    model = hnsw.fit(df)
    
    model.transform(df).write.parquet('/path/to/output', mode='overwrite')


Typically you would want to set numPartitions to the number of executors you have at your disposal


Development
-----------

You can run the integration tests by first running the following command in the root of the project

    mvn clean install 

Followed by running the following commands in the hnswlib-pyspark module

    export SPARK_HOME=/path/to/spark-2.3.0-bin-hadoop2.6
    pip install -e .[test]
    py.test

The easiest way to test changes on a real cluster is to produce an egg file with

    python setup.py bdist_egg
    
And then reference it from spark with

    spark-submit --py-files hnswlib-pyspark/dist/pyspark_hnsw-*.egg YOUR_SCRIPT

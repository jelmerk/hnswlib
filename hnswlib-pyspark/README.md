hnswlib-pyspark
===============

Python wrapper around hnswlib-spark.

To build an egg file run

    python setup.py bdist_egg
    
Then use it as follows

    spark-submit --py-files hnswlib-pyspark/dist/pyspark_hnsw-0.1-*.egg  --jars hnswlib-spark/target/hnswlib-spark-*-jar-with-dependencies.jar YOUR_SCRIPT  
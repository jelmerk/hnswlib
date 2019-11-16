[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-spark_2.3.0_2.11/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-spark_2.3.0_2.11) [![Scaladoc](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-spark_2.3.0_2.11.svg?label=scaladoc)](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-spark_2.3.0_2.11)

hnswlib-spark
=============

[Apache spark](https://spark.apache.org/) integration for hnswlib.

About
-----

The easiest way to use this library with spark is to simply collect your data on the driver node and index it there. 
This does mean you'll have to allocate a lot of cores and memory to the driver.

The alternative to this is to use this moule to shard the index across multiple executors 
and parallelise the indexing / querying. This may be  faster if you have many executors at your disposal and is
appropriate if your dataset won't fit in the driver memory

Setup
-----

Pass the following argument to spark

    --packages 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.23'

Example usage
-------------

    import com.github.jelmerk.spark.knn.hnsw.Hnsw
    
    val hnsw = new Hnsw()
          .setIdentityCol("row_id")
          .setVectorCol("anchor")
          .setNumPartitions(100)
          .setM(64)
          .setEf(5)
          .setEfConstruction(200)
          .setK(5)
          .setDistanceFunction("cosine")
          .setExcludeSelf(false)
          
    val model = hnsw.fit(testData)
    
    model.transform(testData).write.mode(SaveMode.Overwrite).parquet("/path/to/output")

Typically you would want to set numPartitions to the number of executors you have at your disposal

Development
-----------

The easiest way to test changes to the hnswlib codebase is to produce an assembly file with

    mvn clean assembly:assembly
    
And then reference it from spark with

    spark-submit --jars hnswlib-spark/target/hnswlib-spark-*-jar-with-dependencies.jar your.jar

    
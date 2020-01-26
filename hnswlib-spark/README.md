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

    --packages 'com.github.jelmerk:hnswlib-spark_2.3.0_2.11:0.0.29'

Example usage
-------------

    import com.github.jelmerk.spark.knn.hnsw.Hnsw
    import com.github.jelmerk.spark.linalg.Normalizer
    import org.apache.spark.ml.Pipeline
    
    // The cosine distance is obtained with the inner product after normalizing all vectors to unit norm 
    // this is much faster than calculating the cosine distance directly
    
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normalizedFeatures")

    val hnsw = new Hnsw()
      .setIdentityCol("id")
      .setVectorCol("normalizedFeatures")
      .setOutputFormat("minimal")
      .setNumPartitions(2)
      .setM(48)
      .setEf(5)
      .setEfConstruction(200)
      .setK(5)
      .setDistanceFunction("inner-product")
      .setExcludeSelf(true)
    
    val pipeline = new Pipeline()
      .setStages(Array(normalizer, hnsw))

    val model = pipeline.fit(indexItems)

    model.transform(indexItems).write.mode(SaveMode.Overwrite).parquet("/path/to/output")

Suggested configuration
-----------------------

- set `executor.instances` to the same value as the numPartitions property of your Hnsw instance
- set `spark.executor.cores` to as high a value as feasible on your executors while not making your jobs impossible to schedule
- set `spark.task.cpus` to the same value as `spark.executor.cores`
- set `spark.sql.shuffle.partitions` to the same value as `executor.instances`
- set `spark.sql.files.maxpartitionbytes` to 134217728 divided by the value assigned to `executor.instances`
- set `spark.scheduler.minRegisteredResourcesRatio` to `1.0`
- set `spark.speculation` to `false`
- set `spark.driver.memory`: to some arbitrary low value for instance `2g` will do because the model does not run on the driver
- set `spark.executor.memory`: to a value appropriate to the size of your data, typically the will be a large value 
- set `spark.yarn.executor.memoryOverhead` to a value higher than `executorMemory * 0.10` if you get the "Container killed by YARN for exceeding memory limits" error


Note that as it stands increasing the number of partitions will speed up fitting the model but not querying the model.
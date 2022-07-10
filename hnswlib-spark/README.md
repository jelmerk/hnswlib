[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-spark_2.3_2.11/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-spark_2.3_2.11) [![Scaladoc](https://javadoc.io/badge2/com.github.jelmerk/hnswlib-spark_2.3_2.11/javadoc.svg)](https://javadoc.io/doc/com.github.jelmerk/hnswlib-spark_2.3_2.11)


hnswlib-spark
=============

[Apache spark](https://spark.apache.org/) integration for hnswlib.

About
-----

The easiest way to use this library with spark is to simply collect your data on the driver node and index it there.
This does mean you'll have to allocate a lot of cores and memory to the driver.

The alternative to this is to use this module to shard the index across multiple executors
and parallelize the indexing / querying. This may be  faster if you have many executors at your disposal and is
appropriate when your dataset does not fit in the driver memory

Distance functions optimized for use with sparse vectors will automatically be selected base on the input type

Setup
-----

Find the package appropriate for your spark setup

|             | Scala 2.11                                      | Scala 2.12                                      |
| ----------- |-------------------------------------------------|-------------------------------------------------|
| Spark 2.3.x | com.github.jelmerk:hnswlib-spark_2.3_2.11:1.1.0 |                                                 |
| Spark 2.4.x | com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0 | com.github.jelmerk:hnswlib-spark_2.4_2.12:1.1.0 |
| Spark 3.0.x |                                                 | com.github.jelmerk:hnswlib-spark_3.0_2.12:1.1.0 | 
| Spark 3.1.x |                                                 | com.github.jelmerk:hnswlib-spark_3.1_2.12:1.1.0 |
| Spark 3.2.x |                                                 | com.github.jelmerk:hnswlib-spark_3.2_2.12:1.1.0 |
| Spark 3.3.x |                                                 | com.github.jelmerk:hnswlib-spark_3.3_2.12:1.1.0 |

Pass this as an argument to spark

    --packages 'com.github.jelmerk:hnswlib-spark_2.3_2.11:1.1.0'

Example usage
-------------

Basic:

```scala
import com.github.jelmerk.spark.knn.hnsw.HnswSimilarity

val hnsw = new HnswSimilarity()
  .setIdentifierCol("id")
  .setQueryIdentifierCol("id")
  .setFeaturesCol("features")
  .setNumPartitions(2)
  .setM(48)
  .setEf(5)
  .setEfConstruction(200)
  .setK(200)
  .setDistanceFunction("cosine")
  .setExcludeSelf(true)

val model = hnsw.fit(indexItems)

model.transform(indexItems).write.parquet("/path/to/output")
```

Advanced:

```scala
import org.apache.spark.ml.Pipeline

import com.github.jelmerk.spark.knn.bruteforce.BruteForceSimilarity
import com.github.jelmerk.spark.knn.evaluation.KnnSimilarityEvaluator
import com.github.jelmerk.spark.knn.hnsw.HnswSimilarity
import com.github.jelmerk.spark.linalg.Normalizer
import com.github.jelmerk.spark.conversion.VectorConverter

// often it is acceptable to use float instead of double precision. 
// this uses less memory and will be faster 

val converter = new VectorConverter()
    .setInputCol("featuresAsMlLibVector")
    .setOutputCol("features")

// The cosine distance is obtained with the inner product after normalizing all vectors to unit norm 
// this is much faster than calculating the cosine distance directly

val normalizer = new Normalizer()
  .setInputCol("features")
  .setOutputCol("normalizedFeatures")

val hnsw = new HnswSimilarity()
  .setIdentifierCol("id")
  .setQueryIdentifierCol("id")
  .setFeaturesCol("normalizedFeatures")
  .setNumPartitions(2)
  .setK(200)
  .setSimilarityThreshold(0.4)
  .setDistanceFunction("inner-product")
  .setPredictionCol("approximate")
  .setExcludeSelf(true)
  .setM(48)
  .setEfConstruction(200)

val bruteForce = new BruteForceSimilarity()
  .setIdentifierCol(hnsw.getIdentifierCol)
  .setQueryIdentifierCol(hnsw.getQueryIdentifierCol)
  .setFeaturesCol(hnsw.getFeaturesCol)
  .setNumPartitions(2)
  .setK(hnsw.getK)
  .setSimilarityThreshold(hnsw.getSimilarityThreshold)
  .setDistanceFunction(hnsw.getDistanceFunction)
  .setPredictionCol("exact")
  .setExcludeSelf(hnsw.getExcludeSelf)

val pipeline = new Pipeline()
  .setStages(Array(converter, normalizer, hnsw, bruteForce))

val model = pipeline.fit(indexItems)

// computing the exact similarity is expensive so only take a small sample
val queryItems = indexItems.sample(0.01)

val output = model.transform(queryItems)

val evaluator = new KnnSimilarityEvaluator()
  .setApproximateNeighborsCol("approximate")
  .setExactNeighborsCol("exact")

val accuracy = evaluator.evaluate(output)

println(s"Accuracy: $accuracy")

// save the model
model.write.overwrite.save("/path/to/model")
```

Suggested configuration
-----------------------

- set `executor.instances` to the same value as the numPartitions property of your Hnsw instance
- set `spark.executor.cores` to as high a value as feasible on your executors while not making your jobs impossible to schedule
- set `spark.task.cpus` to the same value as `spark.executor.cores`
- set `spark.scheduler.minRegisteredResourcesRatio` to `1.0`
- set `spark.scheduler.maxRegisteredResourcesWaitingTime` to `3600`
- set `spark.speculation` to `false`
- set `spark.dynamicAllocation.enabled` to `false`
- set `spark.task.maxFailures` to `1`
- set `spark.driver.memory`: to some arbitrary low value for instance `2g` will do because the model does not run on the driver
- set `spark.executor.memory`: to a value appropriate to the size of your data, typically this will be a large value
- set `spark.yarn.executor.memoryOverhead` to a value higher than `executorMemory * 0.10` if you get the "Container killed by YARN for exceeding memory limits" error
- set `spark.hnswlib.settings.index.cache_folder` to a folder with plenty of space that you can write to. Defaults to /tmp

Note that as it stands increasing the number of partitions will speed up fitting the model but not querying the model. The only way to speed up querying is by increasing the number of replicas

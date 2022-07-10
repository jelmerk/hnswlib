[![PyPI](https://img.shields.io/pypi/v/pyspark-hnsw)](https://pypi.org/project/pyspark-hnsw/)

hnswlib-pyspark
===============

[PySpark](https://spark.apache.org/) integration for hnswlib.

Setup
-----

Find the package appropriate for your spark setup

|             | Scala 2.11                                      | Scala 2.12                                      |
|-------------|-------------------------------------------------|-------------------------------------------------|
| Spark 2.3.x | com.github.jelmerk:hnswlib-spark_2.3_2.11:1.1.0 |                                                 |
| Spark 2.4.x | com.github.jelmerk:hnswlib-spark_2.4_2.11:1.1.0 | com.github.jelmerk:hnswlib-spark_2.4_2.12:1.1.0 |
| Spark 3.0.x |                                                 | com.github.jelmerk:hnswlib-spark_3.0_2.12:1.1.0 | 
| Spark 3.1.x |                                                 | com.github.jelmerk:hnswlib-spark_3.1_2.12:1.1.0 |
| Spark 3.2.x |                                                 | com.github.jelmerk:hnswlib-spark_3.2_2.12:1.1.0 |
| Spark 3.3.x |                                                 | com.github.jelmerk:hnswlib-spark_3.3_2.12:1.1.0 |


Pass this as an argument to spark

    --packages 'com.github.jelmerk:hnswlib-spark_2.3_2.11:1.1.0'

Then install the python module with

    pip install pyspark-hnsw --upgrade
    

Example usage
-------------

Basic:

```python
from pyspark_hnsw.knn import HnswSimilarity

hnsw = HnswSimilarity(identifierCol='id', featuresCol='features', distanceFunction='cosine', m=48, ef=5, k=200,
                      efConstruction=200, numPartitions=2, excludeSelf=True)

model = hnsw.fit(index_items)

model.transform(index_items).write.parquet('/path/to/output', mode='overwrite')
```

Advanced:

```python
from pyspark.ml import Pipeline
from pyspark_hnsw.evaluation import KnnSimilarityEvaluator
from pyspark_hnsw.knn import *
from pyspark_hnsw.linalg import Normalizer
from pyspark_hnsw.conversion import VectorConverter

# often it is acceptable to use float instead of double precision. 
# this uses less memory and will be faster 
converter = VectorConverter(inputCol='features_as_ml_lib_vector', outputCol='features')

# The cosine distance is obtained with the inner product after normalizing all vectors to unit norm
# this is much faster than calculating the cosine distance directly

normalizer = Normalizer(inputCol='features', outputCol='normalized_features')

hnsw = HnswSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features', distanceFunction='inner-product', m=48, ef=5, k=200,
                      efConstruction=200, numPartitions=2, excludeSelf=True, similarityThreshold=0.4, predictionCol='approximate')
            
brute_force = BruteForceSimilarity(identifierCol='id', queryIdentifierCol='id', featuresCol='normalized_features', distanceFunction='inner-product',
                                   k=200, numPartitions=2, excludeSelf=True, similarityThreshold=0.4, predictionCol='exact')
 
pipeline = Pipeline(stages=[converter, normalizer, hnsw, brute_force])

model = pipeline.fit(index_items)

# computing the exact similarity is expensive so only take a small sample
query_items = index_items.sample(0.01)

output = model.transform(query_items)

evaluator = KnnSimilarityEvaluator(approximateNeighborsCol='approximate', exactNeighborsCol='exact')

accuracy = evaluator.evaluate(output)

print(accuracy)

# save the model
model.write().overwrite().save('/path/to/model')
```

Suggested configuration
-----------------------

- set `executor.instances` to the same value as the numPartitions property of your Hnsw instance
- set `spark.executor.cores` to as high a value as feasible on your executors while not making your jobs impossible to schedule
- set `spark.task.cpus` to the same value as `spark.executor.cores`
- set `spark.scheduler.minRegisteredResourcesRatio` to `1.0`
- set `spark.scheduler.maxRegisteredResourcesWaitingTime` to `3600`
- set `spark.speculation` to `false`
- set `spark.task.maxFailures` to `1`
- set `spark.dynamicAllocation.enabled` to `false`
- set `spark.driver.memory`: to some arbitrary low value for instance `2g` will do because the model does not run on the driver
- set `spark.executor.memory`: to a value appropriate to the size of your data, typically this will be a large value 
- set `spark.yarn.executor.memoryOverhead` to a value higher than `executorMemory * 0.10` if you get the "Container killed by YARN for exceeding memory limits" error
- set `spark.hnswlib.settings.index.cache_folder` to a folder with plenty of space that you can write to. Defaults to /tmp

Note that as it stands increasing the number of partitions will speed up fitting the model but not querying the model. The only way to speed up querying is by increasing the number of replicas


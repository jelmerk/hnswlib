[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-metrics-dropwizard/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-metrics-dropwizard) [![Javadoc](https://javadoc.io/badge2/com.github.jelmerk/hnswlib-metrics-dropwizard/javadoc.svg)](https://javadoc.io/doc/com.github.jelmerk/hnswlib-metrics-dropwizard)

hnswlib-metrics-dropwizard
==========================

[Dropwizard metrics](https://metrics.dropwizard.io) integration for hnswlib.

 
Example usage
-------------


```java
MetricRegistry metricRegistry = SharedMetricRegistries.getDefault();

HnswIndex<String, float[], Word, Float> approximativeIndex = HnswIndex
    .newBuilder(DistanceFunctions.FLOAT_COSINE_DISTANCE, words.size())
        .build();
        
Index<String, float[], Word, Float> groundTruthIndex = approximativeIndex.asExactIndex();

StatisticsDecorator<String, float[], TestItem, Float, HnswIndex<String, float[], Word, Float>, Index<String, float[], Word, Float>> decorator = 
    new StatisticsDecorator<>(metricRegistry, MyClass.class,
        "indexname", approximativeIndex, groundTruthIndex, 1000);
```
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-core) [![Javadoc](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-core.svg?label=javadoc)](http://javadoc-badge.appspot.com/com.github.jelmerk/hnswlib-core)


hnswlib-core
============

Core hnsw library.


Example usage
-------------

```java
import com.github.jelmerk.knn.*;
import com.github.jelmerk.knn.hnsw.*;

HnswIndex<String, float[], Word, Float> index = HnswIndex
    .newBuilder(DistanceFunctions.FLOAT_COSINE_DISTANCE, words.size())
        .withM(10)
        .build();

index.addAll(words);

List<SearchResult<Word, Float>> nearest = index.findNeighbors("king", 10);

for (SearchResult<Word, Float> result : nearest) {
    System.out.println(result.item().id() + " " + result.getDistance());
}
```
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-scala_2.11/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.jelmerk/hnswlib-scala_2.11) [![Scaladoc](https://javadoc.io/badge2/com.github.jelmerk/hnswlib-scala_2.11/javadoc.svg)](https://javadoc.io/doc/com.github.jelmerk/hnswlib-scala_2.11)

hnswlib-scala
=============

[Scala](https://scala-lang.org) wrapper around hnswlib

Example usage
-------------

```scala
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._

val index = HnswIndex[String, Array[Float], Word, Float](floatCosineDistance, words.size, m = 10)
  
index.addAll(words)

index.findNeighbors("king", k = 10).foreach { case SearchResult(item, distance) => 
  println(s"$item $distance")
}
```
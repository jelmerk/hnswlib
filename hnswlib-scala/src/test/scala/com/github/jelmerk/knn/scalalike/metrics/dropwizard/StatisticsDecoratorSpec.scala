package com.github.jelmerk.knn.scalalike.metrics.dropwizard

import com.codahale.metrics.MetricRegistry
import com.github.jelmerk.knn.scalalike.hnsw._
import com.github.jelmerk.knn.scalalike._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import com.codahale.metrics.MetricRegistry.name

class StatisticsDecoratorSpec extends AnyFunSuite {

  test("can construct statistics decorator") {

    val metricRegistry = new MetricRegistry
    val hnswIndex = HnswIndex[String, Array[Float], TestItem, Float](dimensions = 2, floatCosineDistance, maxItemCount = 10)
    val indexName = "indexName"

    val decorator = StatisticsDecorator[String, Array[Float], TestItem, Float,
                      HnswIndex[String, Array[Float], TestItem, Float],
                      Index[String, Array[Float], TestItem, Float]](metricRegistry, classOf[StatisticsDecoratorSpec],
      indexName, hnswIndex, hnswIndex.asExactIndex, 1)

    decorator.get("1")

    metricRegistry.timer(name(getClass, indexName, "get")).getCount should be (1)

  }

}

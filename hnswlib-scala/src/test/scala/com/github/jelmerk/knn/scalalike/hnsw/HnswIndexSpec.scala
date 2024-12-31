package com.github.jelmerk.knn.scalalike.hnsw

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import com.github.jelmerk.knn.JavaObjectSerializer
import com.github.jelmerk.knn.scalalike._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers._

import scala.collection.mutable

class HnswIndexSpec extends AnyFunSuite {

  private val dimensions = 2

  private val item1 = TestItem("1", Array(0.0110f, 0.2341f))
  private val item2 = TestItem("2", Array(0.2300f, 0.3891f))
  private val item3 = TestItem("3", Array(0.4300f, 0.9891f))

  private val k = 10

  test("retrieve m") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, m = 32)
    index.m should be (32)
  }

  test("retrieve ef") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, ef = 100)
    index.ef should be (100)
  }

  test("change ef") {
    val newEfValue = 999
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, ef = 100)
    index.ef = newEfValue
    index.ef should be (newEfValue)
  }

  test("retrieve efConstruction") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, efConstruction = 100)
    index.efConstruction should be (100)
  }

  test("retrieve maxItemCount") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.maxItemCount should be (10)
  }

  test("retrieve distanceFunction") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.distanceFunction should be (floatCosineDistance)
  }

  test("retrieve removeEnabled") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, removeEnabled = true)
    index.removeEnabled should be (true)
  }

  test("retrieve distance ordering") {
    val ordering = Ordering[Float]
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)(ordering)
    index.distanceOrdering should be theSameInstanceAs ordering
  }

  test("retrieve itemIdSerializer") {
    val itemIdSerializer = new JavaObjectSerializer[String]
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, itemIdSerializer = itemIdSerializer)
    index.itemIdSerializer should be theSameInstanceAs itemIdSerializer
  }

  test("retrieve itemSerializer") {
    val itemSerializer = new JavaObjectSerializer[TestItem]
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, itemSerializer = itemSerializer)
    index.itemSerializer should be theSameInstanceAs itemSerializer
  }

  test("optionally get non-existent item from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.get("1") should be (None)
  }

  test("optionally get existing item from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.add(item1)
    index.get(item1.id) should be (Some(item1))
  }

  test("get existing item from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.add(item1)
    index(item1.id) should be (item1)
  }

  test("get non-existent item from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    an[NoSuchElementException] shouldBe thrownBy {
      index(item1.id) should be (item1)
    }
  }

  test("check if item is contained in index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)

    index.contains(item1.id) should be (false)
    index.add(item1)
    index.contains(item1.id) should be (true)
  }

  test("get items from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.add(item1)

    index.toSeq should be (Seq(item1))
  }

  test("retrieve size of index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)

    index.add(item1)
    index.size should be (1)
  }

  test("remove item from index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10, removeEnabled = true)

    index.add(item1)
    index.size should be (1)
    index.remove(item1.id, item1.version) should be (true)
    index.size should be (0)
  }

  test("find nearest") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.addAll(Seq(item1, item2, item3))

    val results = index.findNearest(item1.vector, k)

    results should contain inOrderOnly (
      SearchResult(item1, 0f),
      SearchResult(item3, 0.06521261f),
      SearchResult(item2, 0.11621308f)
    )
  }

  test("find neighbors") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.addAll(Seq(item1, item2, item3))

    val results = index.findNeighbors(item1.id, k)

    results should contain inOrderOnly (
      SearchResult(item3, 0.06521261f),
      SearchResult(item2, 0.11621308f)
    )
  }

  test("calls progress listener when indexing") {
    val updates = mutable.ArrayBuffer.empty[ProgressUpdate]

    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.addAll(Seq(item1, item2, item3), progressUpdateInterval = 2, numThreads = 1,
      listener = (workDone, max) => updates += ProgressUpdate(workDone, max))

    updates should contain inOrderOnly (
      ProgressUpdate(2, 3),
      ProgressUpdate(3, 3) // last item always emits
    )
  }

  test("can load saved index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.add(item1)

    val baos = new ByteArrayOutputStream

    index.save(baos)

    val loaded = HnswIndex.loadFromInputStream[String, Array[Float], TestItem, Float](new ByteArrayInputStream(baos.toByteArray))

    loaded.size should be (1)
  }

  test("can create exact view on hnsw index") {
    val index = HnswIndex[String, Array[Float], TestItem, Float](dimensions, floatCosineDistance, maxItemCount = 10)
    index.add(item1)

    index.asExactIndex.size should be (1)
  }

  test("creates an empty immutable index") {
    val index = HnswIndex.empty[String, Array[Float], TestItem, Float]
    index.size should be (0)
  }

}

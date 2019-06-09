import java.io.File

import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._

import scala.io.Source
import scala.util.Random

import com.github.jelmerk.knn.DistanceFunctions

case class FastTextWord(id: String, vector: Array[Float], expired: Boolean)
  extends Item[String, Array[Float]]

object Testje {

  val random = new Random(42)

  val expiredPercent = 50
  val numSearchResults = 100
  val numSamples = 200
  val indexSize = 1000000

//  val m = 48
  val m = 64
  val ef = 200
  val efConstruction = 200

  def main(args: Array[String]): Unit = {

    val words = Source.fromFile(new File("/Users/jkuperus/Downloads/cc.nl.300.vec"))
      .getLines()
      .slice(1, indexSize + 1)
      .map { line =>
        val tokens = line.split(" ")
        val word = tokens.head
        val vector = tokens.tail.map(_.toFloat)

        val expired = random.nextInt(100) > expiredPercent

        FastTextWord(word, vector, expired)
      }
      .toSeq

    val fullHnswIndex =
      HnswIndex[String, Array[Float], FastTextWord, Float](DistanceFunctions.cosineDistance, words.size, m, ef, efConstruction)

    fullHnswIndex.addAll(words, listener = (workDone: Int, max: Int) => {
      println(s"Indexed $workDone of $max items for full hnsw index.")
    })

    val fullBruteForceIndex = fullHnswIndex.exactView

    val nonExpiredWords = words.filterNot(_.expired)

    val nonExpiredHnswIndex =
      HnswIndex[String, Array[Float], FastTextWord, Float](DistanceFunctions.cosineDistance, words.size, m, ef, efConstruction)

    nonExpiredHnswIndex.addAll(nonExpiredWords, listener = (workDone: Int, max: Int) => {
      println(s"Indexed $workDone of $max items for non expired words hnsw index.")
    })

    val randomExpiredWords = takeRandomExpiredWords(words, numSamples)

    val avgPrecisionFull  = randomExpiredWords.map { word =>
      val expected = fullBruteForceIndex.findNearest(word.vector, numSearchResults).filterNot(_.item.expired)
      val actual = fullHnswIndex.findNearest(word.vector, numSearchResults).filterNot(_.item.expired)

      calculatePrecision(expected, actual)
    }.sum / randomExpiredWords.size

    println(s"precision on full index $avgPrecisionFull")


    val avgPrecisionSplit = randomExpiredWords.map { word =>
      val expected = fullBruteForceIndex.findNearest(word.vector, numSearchResults).filterNot(_.item.expired)
      val actual = nonExpiredHnswIndex.findNearest(word.vector, numSearchResults)

      calculatePrecision(expected, actual)
    }.sum / randomExpiredWords.size

    println(s"precision on split index $avgPrecisionSplit")

  }

  def takeRandomExpiredWords(words: Seq[FastTextWord], k: Int): Seq[FastTextWord] =
    Iterator.continually(words(random.nextInt(words.size)))
        .filter(_.expired)
        .take(k)
        .toSeq

  def calculatePrecision(expected: Seq[SearchResult[FastTextWord, Float]],
                         actual: Seq[SearchResult[FastTextWord, Float]]): Double = {

    val correct = actual.map(expected.contains).count(identity)
    correct.toDouble / expected.size.toDouble
  }

}

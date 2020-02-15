package com.github.jelmerk.knn.examples

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

import scala.io.{Codec, Source, StdIn}

import com.github.jelmerk.knn.examples.IoUtils._
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._

@SerialVersionUID(1L) case class Word(id: String, vector: Array[Float]) extends Item[String, Array[Float]] {
  override def dimensions(): Int = vector.length
}

/**
  * Example application that downloads the english fast-text word vectors, inserts them into an hnsw index and lets
  * you query them.
  */
object FastText extends App {

  val url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"

  val inputFile = Paths.get(sys.props("java.io.tmpdir"), "cc.en.300.vec.gz")

  for {
    inputFilePath <- Some(inputFile).filter(f => Files.exists(f)).orElse(Some(downloadFile(url, inputFile)))
    words <- Some(loadWords(inputFilePath))
  } {
    println("Constructing index.")

    val hnswIndex = HnswIndex[String, Array[Float], Word, Float](dimensions = 300, floatCosineDistance, words.size, m = 16,  ef = 200, efConstruction = 200)

    hnswIndex.addAll(words, listener = (workDone: Int, max: Int) =>
      println(s"Added $workDone out of $max words to the index.")
    )

    val groundTruthIndex = hnswIndex.asExactIndex

    val k = 10

    while(true) {
      println("Enter an english word : ")

      val input = StdIn.readLine()

      println("Most similar words found using HNSW index :\n")

      val approximateResults = hnswIndex.findNeighbors(input, k)

      approximateResults.foreach { case SearchResult(word, distance) =>
        println(f"$word $distance#.4f")
      }

      println("\nMost similar words found using exact index :\n")

      val groundTruthResults = groundTruthIndex.findNeighbors(input, k)

      groundTruthResults.foreach { case SearchResult(word, distance) =>
        println(f"$word $distance#.4f")
      }

      val accuracy = groundTruthResults.count(approximateResults.contains) / groundTruthResults.size.toDouble

      println(f"%nAccuracy : $accuracy#.4f%n")
    }
  }

  def loadWords(path: Path) = {
    println(s"Loading words from $path")

    withClosableResource(Source.fromInputStream(new GZIPInputStream(Files.newInputStream(path)))(Codec.UTF8)) { source =>
      source.getLines()
        .drop(1)
        .map(_.split(" "))
        .map { tokens => Word(tokens.head, tokens.tail.map(_.toFloat)) }
        .toList
    }
  }

  def downloadFile(url: String, file: Path) = {
    println(s"Downloading $url to $file. This may take a while.")

    withClosableResource(new URL(url).openStream) { in =>
      Files.copy(in, file)
      file
    }
  }

}

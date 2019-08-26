package com.github.jelmerk.knn.examples

import java.net.URL
import java.nio.file.{Files, Path, Paths}
import java.util.zip.GZIPInputStream

import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.scalalike.hnsw._

import scala.io.{Codec, Source}

case class Word(id: String, vector: Array[Float]) extends Item[String, Array[Float]]

object FastText extends App {

  val url: String = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"

  val inputFile: Path = Paths.get(sys.props("java.io.tmpdir"), "cc.en.300.vec.gz")

  for {
    inputFilePath <- Some(inputFile).filter(f => Files.exists(f)).orElse(Some(downloadFile(url, inputFile)))
    words <- Some(loadWords(inputFilePath))
  } {

    println("Constructing index.")

    val index = HnswIndex[String, Array[Float], Word, Float](floatCosineDistance, words.size, m = 16)

    index.addAll(words, listener = (workDone: Int, max: Int) => {
      println(s"Added $workDone out of $max words to the index.")
    })

    index.findNeighbors("bike", k = 10).foreach { case SearchResult(word, distance) =>
      println(f"$word $distance#.4f")
    }
  }

  def loadWords(path: Path): Seq[Word] = {
    println(s"Loading words from $path")

    val source = Source.fromInputStream(new GZIPInputStream(Files.newInputStream(path)))(Codec.UTF8)
    try {
      source.getLines()
        .drop(1)
        .map(_.split(" "))
        .map { tokens => Word(tokens.head, tokens.tail.map(_.toFloat)) }
        .toList
    } finally {
      source.close()
    }
  }

  def downloadFile(url: String, file: Path): Path = {
    println(s"Downloading $url to $file. This may take a while.")

    val in = new URL(url).openStream
    try {
      Files.copy(in, file)
    } finally {
      in.close()
    }
    file
  }

}

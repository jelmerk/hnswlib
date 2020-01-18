package com.github.jelmerk.spark.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.SequenceFileInputFormat

/**
  * Extends SequenceFileInputFormat to make it not splittable.
  */
private[spark] class UnsplittableSequenceFileInputFormat[K, V] extends SequenceFileInputFormat[K, V] {
  override def isSplitable(fs: FileSystem, filename: Path): Boolean = false
}

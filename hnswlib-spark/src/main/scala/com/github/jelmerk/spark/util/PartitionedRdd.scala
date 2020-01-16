package com.github.jelmerk.spark.util

import org.apache.spark.rdd.RDD
import org.apache.spark.{Partition, Partitioner, TaskContext}

import scala.reflect.ClassTag

/**
  * When an rdd gets saved to disk it loses it's partitioner. This wrapper remedies this.
  *
  * @param prev the rdd to delegate to
  * @param partitioner the partitioner to assign to the RDD
  *
  * @tparam T type of element in the RDD
  */
private[spark] class PartitionedRdd[T: ClassTag](prev: RDD[T], override val partitioner: Option[Partitioner]) extends RDD[T](prev) {

  override def compute(split: Partition, context: TaskContext): Iterator[T] = firstParent.compute(split, context)

  override protected def getPartitions: Array[Partition] = firstParent.partitions

}

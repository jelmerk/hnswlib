package com.github.jelmerk.spark

import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.serializers.JavaSerializer
import com.github.jelmerk.knn.scalalike.hnsw.HnswIndex
import com.github.jelmerk.spark.util.SerializableConfiguration
import org.apache.spark.serializer.KryoRegistrator

/**
 * Implementation of KryoRegistrator that registers hnswlib classes with spark.
 * Can be registered by setting spark.kryo.registrator to com.github.jelmerk.spark.HnswLibKryoRegistrator
 */
class HnswLibKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo): Unit = {
    kryo.register(classOf[HnswIndex[_, _, _, _]], new JavaSerializer)
    kryo.register(classOf[SerializableConfiguration], new JavaSerializer)
  }
}

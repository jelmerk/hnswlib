package com.github.jelmerk.spark.knn

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}

import com.github.jelmerk.knn.scalalike.{Item, ObjectSerializer}
import org.apache.spark.ml.linalg.{Vectors, Vector}
import org.scalatest.FunSuite
import org.scalatest.Matchers._

class SerializerSpec extends FunSuite {

  test("serialize objects") {
    validateSerializability(IntSerializer, 1)
    validateSerializability(LongSerializer, 1L)
    validateSerializability(StringSerializer, "1")
    validateSerializability(VectorSerializer, Vectors.dense(Array(0.1, 0.2)))
    validateSerializability(VectorSerializer, Vectors.sparse(3, Array(0, 1), Array(0.1, 0.2)))
    validateSerializability(FloatArraySerializer, Array(0.1f, 0.2f))
    validateSerializability(DoubleArraySerializer, Array(0.1, 0.2))
    validateSerializability(IntDoubleArrayIndexItemSerializer, IntDoubleArrayIndexItem(1, Array(0.1, 0.2)), deepCompare[Int, Array[Double]])
    validateSerializability(LongDoubleArrayIndexItemSerializer, LongDoubleArrayIndexItem(1L, Array(0.1, 0.2)), deepCompare[Long, Array[Double]])
    validateSerializability(StringDoubleArrayIndexItemSerializer, StringDoubleArrayIndexItem("1", Array(0.1, 0.2)), deepCompare[String, Array[Double]])

    validateSerializability(IntFloatArrayIndexItemSerializer, IntFloatArrayIndexItem(1, Array(0.1f, 0.2f)), deepCompare[Int, Array[Float]])
    validateSerializability(LongFloatArrayIndexItemSerializer, LongFloatArrayIndexItem(1L, Array(0.1f, 0.2f)), deepCompare[Long, Array[Float]])
    validateSerializability(StringFloatArrayIndexItemSerializer, StringFloatArrayIndexItem("1", Array(0.1f, 0.2f)), deepCompare[String, Array[Float]])

    validateSerializability(IntVectorIndexItemSerializer, IntVectorIndexItem(1, Vectors.dense(0.1, 0.2)), deepCompare[Int, Vector])
    validateSerializability(LongVectorIndexItemSerializer, LongVectorIndexItem(1L, Vectors.dense(0.1, 0.2)), deepCompare[Long, Vector])
    validateSerializability(StringVectorIndexItemSerializer, StringVectorIndexItem("1", Vectors.dense(0.1, 0.2)), deepCompare[String, Vector])
  }

  private def validateSerializability[T](serializer: ObjectSerializer[T], value: T,
                                         validation: (T, T) => Unit = simpleCompare[T] _): Unit = {
    val baos = new ByteArrayOutputStream
    val oos = new ObjectOutputStream(baos)
    serializer.write(value, oos)

    oos.flush()

    val bais = new ByteArrayInputStream(baos.toByteArray)
    val ois = new ObjectInputStream(bais)
    val read = serializer.read(ois)

    validation(read, value)
  }

  private def simpleCompare[T](in: T, out: T): Unit = {
    in should be (out)
  }

  private def deepCompare[TId, TVector](in: Item[TId, TVector], out: Item[TId, TVector]): Unit  = {
    in.vector should be(out.vector)
    in.id should be (out.id)
  }

}

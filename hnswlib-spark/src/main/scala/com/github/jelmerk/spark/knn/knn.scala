package com.github.jelmerk.spark

import java.io.{ObjectInput, ObjectOutput}

import com.github.jelmerk.knn.scalalike.ObjectSerializer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}

package object knn {

  implicit object StringSerializer extends ObjectSerializer[String] {
    override def write(item: String, out: ObjectOutput): Unit = out.writeUTF(item)
    override def read(in: ObjectInput): String = in.readUTF()
  }

  implicit object IntSerializer extends ObjectSerializer[Int] {
    override def write(item: Int, out: ObjectOutput): Unit = out.writeInt(item)
    override def read(in: ObjectInput): Int = in.readInt()
  }

  implicit object LongSerializer extends ObjectSerializer[Long] {
    override def write(item: Long, out: ObjectOutput): Unit = out.writeLong(item)
    override def read(in: ObjectInput): Long = in.readLong()
  }

  implicit object FloatArraySerializer extends ObjectSerializer[Array[Float]] {
    override def write(item: Array[Float], out: ObjectOutput): Unit = {
      out.writeInt(item.length)
      item.foreach(out.writeFloat)
    }

    override def read(in: ObjectInput): Array[Float] = {
      val length = in.readInt()
      val item = Array.ofDim[Float](length)

      for (i <- 0 until length) {
        item(i) = in.readFloat()
      }
      item
    }
  }

  implicit object DoubleArraySerializer extends ObjectSerializer[Array[Double]] {
    override def write(item: Array[Double], out: ObjectOutput): Unit = {
      out.writeInt(item.length)
      item.foreach(out.writeDouble)
    }

    override def read(in: ObjectInput): Array[Double] = {
      val length = in.readInt()
      val item = Array.ofDim[Double](length)

      for (i <- 0 until length) {
        item(i) = in.readDouble()
      }
      item
    }
  }


  implicit object VectorSerializer extends ObjectSerializer[Vector] {
    override def write(item: Vector, out: ObjectOutput): Unit = item match {
      case v: DenseVector =>
        out.writeBoolean(true)
        out.writeInt(v.size)
        v.values.foreach(out.writeDouble)

      case v: SparseVector =>
        out.writeBoolean(false)
        out.writeInt(v.size)
        out.writeInt(v.indices.length)
        v.indices.foreach(out.writeInt)
        v.values.foreach(out.writeDouble)
    }

    override def read(in: ObjectInput): Vector = {
      val isDense = in.readBoolean()
      val size = in.readInt()

      if (isDense) {
        val values = Array.ofDim[Double](size)

        for (i <- 0 until size) {
          values(i) = in.readDouble()
        }

        Vectors.dense(values)
      } else {
        val numFilled = in.readInt()
        val indices = Array.ofDim[Int](numFilled)

        for (i <- 0 until numFilled) {
          indices(i) = in.readInt()
        }

        val values = Array.ofDim[Double](numFilled)

        for (i <- 0 until numFilled) {
          values(i) = in.readDouble()
        }

        Vectors.sparse(size, indices, values)
      }
    }
  }


  implicit object IntVectorIndexItemSerializer extends ObjectSerializer[IntVectorIndexItem] {
    override def write(item: IntVectorIndexItem, out: ObjectOutput): Unit = {
      IntSerializer.write(item.id, out)
      VectorSerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): IntVectorIndexItem = {
      val id = IntSerializer.read(in)
      val vector = VectorSerializer.read(in)
      IntVectorIndexItem(id, vector)
    }
  }

  implicit object LongVectorIndexItemSerializer extends ObjectSerializer[LongVectorIndexItem] {
    override def write(item: LongVectorIndexItem, out: ObjectOutput): Unit = {
      LongSerializer.write(item.id, out)
      VectorSerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): LongVectorIndexItem = {
      val id = LongSerializer.read(in)
      val vector = VectorSerializer.read(in)
      LongVectorIndexItem(id, vector)
    }
  }

  implicit object StringVectorIndexItemSerializer extends ObjectSerializer[StringVectorIndexItem] {
    override def write(item: StringVectorIndexItem, out: ObjectOutput): Unit = {
      StringSerializer.write(item.id, out)
      VectorSerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): StringVectorIndexItem = {
      val id = StringSerializer.read(in)
      val vector = VectorSerializer.read(in)
      StringVectorIndexItem(id, vector)
    }
  }


  implicit object IntFloatArrayIndexItemSerializer extends ObjectSerializer[IntFloatArrayIndexItem] {
    override def write(item: IntFloatArrayIndexItem, out: ObjectOutput): Unit = {
      IntSerializer.write(item.id, out)
      FloatArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): IntFloatArrayIndexItem = {
      val id = IntSerializer.read(in)
      val vector = FloatArraySerializer.read(in)
      IntFloatArrayIndexItem(id, vector)
    }
  }

  implicit object LongFloatArrayIndexItemSerializer extends ObjectSerializer[LongFloatArrayIndexItem] {
    override def write(item: LongFloatArrayIndexItem, out: ObjectOutput): Unit = {
      LongSerializer.write(item.id, out)
      FloatArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): LongFloatArrayIndexItem = {
      val id = LongSerializer.read(in)
      val vector = FloatArraySerializer.read(in)
      LongFloatArrayIndexItem(id, vector)
    }
  }

  implicit object StringFloatArrayIndexItemSerializer extends ObjectSerializer[StringFloatArrayIndexItem] {
    override def write(item: StringFloatArrayIndexItem, out: ObjectOutput): Unit = {
      StringSerializer.write(item.id, out)
      FloatArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): StringFloatArrayIndexItem = {
      val id = StringSerializer.read(in)
      val vector = FloatArraySerializer.read(in)
      StringFloatArrayIndexItem(id, vector)
    }
  }


  implicit object IntDoubleArrayIndexItemSerializer extends ObjectSerializer[IntDoubleArrayIndexItem] {
    override def write(item: IntDoubleArrayIndexItem, out: ObjectOutput): Unit = {
      IntSerializer.write(item.id, out)
      DoubleArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): IntDoubleArrayIndexItem = {
      val id = IntSerializer.read(in)
      val vector = DoubleArraySerializer.read(in)
      IntDoubleArrayIndexItem(id, vector)
    }
  }

  implicit object LongDoubleArrayIndexItemSerializer extends ObjectSerializer[LongDoubleArrayIndexItem] {
    override def write(item: LongDoubleArrayIndexItem, out: ObjectOutput): Unit = {
      LongSerializer.write(item.id, out)
      DoubleArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): LongDoubleArrayIndexItem = {
      val id = LongSerializer.read(in)
      val vector = DoubleArraySerializer.read(in)
      LongDoubleArrayIndexItem(id, vector)
    }
  }

  implicit object StringDoubleArrayIndexItemSerializer extends ObjectSerializer[StringDoubleArrayIndexItem] {
    override def write(item: StringDoubleArrayIndexItem, out: ObjectOutput): Unit = {
      StringSerializer.write(item.id, out)
      DoubleArraySerializer.write(item.vector, out)
    }

    override def read(in: ObjectInput): StringDoubleArrayIndexItem = {
      val id = StringSerializer.read(in)
      val vector = DoubleArraySerializer.read(in)
      StringDoubleArrayIndexItem(id, vector)
    }
  }

}

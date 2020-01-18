package com.github.jelmerk.spark.util

import java.io.{ByteArrayInputStream, ObjectInputStream}

/**
  * Methods copied from import org.apache.spark.util.Utils because they are not accessible
  */
private[spark] object Utils {

  /**
    * Deserialize an object using Java serialization
    **/
  def deserialize[T](bytes: Array[Byte]): T = {
    val bis = new ByteArrayInputStream(bytes)
    val ois = new ObjectInputStream(bis)
    ois.readObject.asInstanceOf[T]
  }

}

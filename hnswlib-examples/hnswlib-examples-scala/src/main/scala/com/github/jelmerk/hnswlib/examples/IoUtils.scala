package com.github.jelmerk.hnswlib.examples

import scala.language.reflectiveCalls

object IoUtils {

  def withClosableResource[A, B <: {def close(): Unit}] (closeable: B) (f: B => A): A = {
    try {
      f(closeable)
    } finally {
      closeable.close()
    }
  }

}
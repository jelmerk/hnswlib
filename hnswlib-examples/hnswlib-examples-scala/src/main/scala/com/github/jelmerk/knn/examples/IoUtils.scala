package com.github.jelmerk.knn.examples

object IoUtils {

  def withClosableResource[A, B <: AutoCloseable] (closeable: B) (f: B => A): A = {
    try {
      f(closeable)
    } finally {
      closeable.close()
    }
  }

}
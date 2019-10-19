package com.github.jelmerk.knn.scalalike

import com.github.jelmerk.knn.{ProgressListener => JProgressListener}

/**
  * Adapts the interface of a java progress listener to that of a scala progress listener
  *
  * @param delegate the java progress listener to delegate to
  */
@SerialVersionUID(1L)
private[scalalike] class JavaProgressListenerAdapter(val delegate: JProgressListener) extends ProgressListener {

  override def apply(workDone: Int, max: Int): Unit = {
    delegate.updateProgress(workDone, max)
  }
}
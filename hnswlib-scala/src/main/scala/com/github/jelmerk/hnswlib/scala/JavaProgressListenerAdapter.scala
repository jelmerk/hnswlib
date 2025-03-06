package com.github.jelmerk.hnswlib.scala

import com.github.jelmerk.hnswlib.core.{ProgressListener => JProgressListener}

/**
  * Adapts the interface of a java progress listener to that of a scala progress listener
  *
  * @param delegate the java progress listener to delegate to
  */
@SerialVersionUID(1L)
private[scala] class JavaProgressListenerAdapter(val delegate: JProgressListener) extends ProgressListener {

  override def apply(workDone: Int, max: Int): Unit = {
    delegate.updateProgress(workDone, max)
  }
}
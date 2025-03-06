package com.github.jelmerk.hnswlib.scala

import com.github.jelmerk.hnswlib.core.{ProgressListener => JProgressListener}
/**
  * Adapts the interface of a scala progress listener to that of a java progress listener
  *
  * @param delegate the scala progress listener to delegate to
  */
@SerialVersionUID(1L)
private[scala] class ScalaProgressListenerAdapter(val delegate: ProgressListener) extends JProgressListener {

  override def updateProgress(workDone: Int, max: Int): Unit = delegate(workDone, max)
}
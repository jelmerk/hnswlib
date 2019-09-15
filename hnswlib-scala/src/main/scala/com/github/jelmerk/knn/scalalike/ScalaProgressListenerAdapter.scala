package com.github.jelmerk.knn.scalalike

import com.github.jelmerk.knn.{ProgressListener => JProgressListener}
/**
  * Adapts the interface of a scala progress listener to that of a java progress listener
  *
  * @param delegate the scala progress listener to delegate to
  */
@SerialVersionUID(1L)
class ScalaProgressListenerAdapter(val delegate: ProgressListener) extends JProgressListener {

  /**
    * @inheritdoc
    */
  override def updateProgress(workDone: Int, max: Int): Unit = delegate(workDone, max)
}
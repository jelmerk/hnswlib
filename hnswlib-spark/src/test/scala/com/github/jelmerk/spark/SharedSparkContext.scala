package com.github.jelmerk.spark

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.scalatest.{BeforeAndAfterAll, Suite}

/** Shares a local `SparkContext` between all tests in a suite and closes it at the end */
trait SharedSparkContext extends BeforeAndAfterAll {
  self: Suite =>

  @transient private var sparkSession: SparkSession = _

  def appID: String = this.getClass.getName + math.floor(math.random * 10E4).toLong.toString

  def conf: SparkConf = {
    new SparkConf().
      setMaster("local[*]").
      setAppName("test").
      set("spark.ui.enabled", "false").
      set("spark.app.id", appID).
      set("spark.driver.host", "localhost")
  }

  def spark: SQLContext = sparkSession.sqlContext

  override def beforeAll(): Unit = {
    sparkSession = SparkSession.builder().config(conf).getOrCreate()
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    try {
      Option(sparkSession).foreach { _.stop() }
      // To avoid Akka rebinding to the same port, since it doesn't
      // unbind immediately on shutdown.
      System.clearProperty("spark.driver.port")
      sparkSession = null
    } finally {
      super.afterAll()
    }
  }

}
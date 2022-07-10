package com.github.jelmerk.spark.knn

import java.io.InputStream
import java.net.InetAddress
import java.util.concurrent.{CountDownLatch, ExecutionException, FutureTask, LinkedBlockingQueue, ThreadLocalRandom, ThreadPoolExecutor, TimeUnit}
import com.github.jelmerk.knn.ObjectSerializer

import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try
import org.apache.hadoop.fs.{FileUtil, Path}
import org.apache.spark.{Partitioner, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.apache.spark.scheduler.{SparkListener, SparkListenerApplicationEnd}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.JsonDSL._
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.knn.util.NamedThreadFactory
import com.github.jelmerk.spark.linalg.functions.VectorDistanceFunctions
import com.github.jelmerk.spark.util.SerializableConfiguration
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder

import scala.annotation.tailrec
import scala.util.control.NonFatal


private[knn] case class IntDoubleArrayIndexItem(id: Int, vector: Array[Double]) extends Item[Int, Array[Double]] {
  override def dimensions: Int = vector.length
}

private[knn] case class LongDoubleArrayIndexItem(id: Long, vector: Array[Double]) extends Item[Long, Array[Double]] {
  override def dimensions: Int = vector.length
}

private[knn] case class StringDoubleArrayIndexItem(id: String, vector: Array[Double]) extends Item[String, Array[Double]] {
  override def dimensions: Int = vector.length
}


private[knn] case class IntFloatArrayIndexItem(id: Int, vector: Array[Float]) extends Item[Int, Array[Float]] {
  override def dimensions: Int = vector.length
}

private[knn] case class LongFloatArrayIndexItem(id: Long, vector: Array[Float]) extends Item[Long, Array[Float]] {
  override def dimensions: Int = vector.length
}

private[knn] case class StringFloatArrayIndexItem(id: String, vector: Array[Float]) extends Item[String, Array[Float]] {
  override def dimensions: Int = vector.length
}


private[knn] case class IntVectorIndexItem(id: Int, vector: Vector) extends Item[Int, Vector] {
  override def dimensions: Int = vector.size
}

private[knn] case class LongVectorIndexItem(id: Long, vector: Vector) extends Item[Long, Vector] {
  override def dimensions: Int = vector.size
}

private[knn] case class StringVectorIndexItem(id: String, vector: Vector) extends Item[String, Vector] {
  override def dimensions: Int = vector.size
}

/**
  * Neighbor of an item.
  *
  * @param neighbor identifies the neighbor
  * @param distance distance to the item
  *
  * @tparam TId type of the index item identifier
  * @tparam TDistance type of distance
  */
private[knn] case class Neighbor[TId, TDistance] (neighbor: TId, distance: TDistance)

/**
  * Common params for KnnAlgorithm and KnnModel.
  */
private[knn] trait KnnModelParams extends Params with HasFeaturesCol with HasPredictionCol {

  /**
    * Param for the column name for the query identifier.
    *
    * @group param
    */
  final val queryIdentifierCol = new Param[String](this, "queryIdentifierCol", "column name for the query identifier")

  /** @group getParam */
  final def getQueryIdentifierCol: String = $(queryIdentifierCol)

  /**
   * Param for the column name for the query partitions.
   *
   * @group param
   */
  final val queryPartitionsCol = new Param[String](this, "queryPartitionsCol", "column name for the query partitions")

  /** @group getParam */
  final def getQueryPartitionsCol: String = $(queryPartitionsCol)

  /**
    * Param for number of neighbors to find (> 0).
    * Default: 5
    *
    * @group param
    */
  final val k = new IntParam(this, "k", "number of neighbors to find", ParamValidators.gt(0))

  /** @group getParam */
  final def getK: Int = $(k)

  /**
    * Param that indicates whether to not return the a candidate when it's identifier equals the query identifier
    * Default: false
    *
    * @group param
    */
  final val excludeSelf = new BooleanParam(this, "excludeSelf", "whether to include the row identifier as a candidate neighbor")

  /** @group getParam */
  final def getExcludeSelf: Boolean = $(excludeSelf)

  /**
    * Param for the threshold value for inclusion. -1 indicates no threshold
    * Default: -1
    *
    * @group param
    */
  final val similarityThreshold = new DoubleParam(this, "similarityThreshold", "do not return neighbors further away than this distance")

  /** @group getParam */
  final def getSimilarityThreshold: Double = $(similarityThreshold)

  /**
    * Param that specifies the number of index replicas to create when querying the index. More replicas means you can
    * execute more queries in parallel at the expense of increased resource usage.
    * Default: 0
    *
    * @group param
    */
  final val numReplicas = new IntParam(this, "numReplicas", "number of index replicas to create when querying")

  /** @group getParam */
  final def getNumReplicas: Int = $(numReplicas)

  /**
    * Param that specifies the number of threads to use.
    * Default: number of processors available to the Java virtual machine
    *
    * @group param
    */
  final val parallelism = new IntParam(this, "parallelism", "number of threads to use")

  /** @group getParam */
  final def getParallelism: Int = $(parallelism)

  /**
    * Param for the output format to produce. One of "full", "minimal" Setting this to minimal is more efficient
    * when all you need is the identifier with its neighbors
    *
    * Default: "full"
    *
    * @group param
    */
  final val outputFormat = new Param[String](this, "outputFormat", "output format to produce")

  /** @group getParam */
  final def getOutputFormat: String = $(outputFormat)

  setDefault(k -> 5, predictionCol -> "prediction", featuresCol -> "features",
    excludeSelf -> false, similarityThreshold -> -1, outputFormat -> "full")

  protected def validateAndTransformSchema(schema: StructType, identifierDataType: DataType): StructType = {

    val distanceType = schema(getFeaturesCol).dataType match {
      case ArrayType(FloatType, _) => FloatType
      case _ => DoubleType
    }

    val predictionStruct = new StructType()
      .add("neighbor", identifierDataType, nullable = false)
      .add("distance", distanceType, nullable = false)

    val neighborsField = StructField(getPredictionCol, new ArrayType(predictionStruct, containsNull = false))

    getOutputFormat match {
      case "minimal" if !isSet(queryIdentifierCol) => throw new IllegalArgumentException("queryIdentifierCol must be set when using outputFormat minimal.")
      case "minimal" =>
        new StructType()
          .add(schema(getQueryIdentifierCol))
          .add(neighborsField)
      case _ =>
        if (schema.fieldNames.contains(getPredictionCol)) {
          throw new IllegalArgumentException(s"Output column $getPredictionCol already exists.")
        }
        schema
          .add(neighborsField)
    }
  }
}

/**
  * Params for knn algorithms.
  */
private[knn] trait KnnAlgorithmParams extends KnnModelParams {

  /**
    * Param for the column name for the row identifier.
    * Default: "id"
    *
    * @group param
    */
  final val identifierCol = new Param[String](this, "identifierCol", "column name for the row identifier")

  /** @group getParam */
  final def getIdentifierCol: String = $(identifierCol)

  /**
    * Number of partitions (default: 1)
    */
  final val numPartitions = new IntParam(this, "numPartitions",
    "number of partitions", ParamValidators.gt(0))

  /** @group getParam */
  final def getNumPartitions: Int = $(numPartitions)

  /**
    * Param for the distance function to use. One of "bray-curtis", "canberra",  "cosine", "correlation", "euclidean",
    * "inner-product", "manhattan" or the fully qualified classname of a distance function
    * Default: "cosine"
    *
    * @group param
    */
  final val distanceFunction = new Param[String](this, "distanceFunction", "distance function to use")

  /** @group getParam */
  final def getDistanceFunction: String = $(distanceFunction)

  /**
   * Param for the partition identifier
   */
  final val partitionCol = new Param[String](this, "partitionCol", "column name for the partition identifier")

  /** @group getParam */
  final def getPartitionCol: String = $(partitionCol)

  /**
   * Param to the initial model. All the vectors from the initial model will included in the final output model.
   */
  final val initialModelPath = new Param[String](this, "initialModelPath", "path to the initial model")

  /** @group getParam */
  final def getInitialModelPath: String = $(initialModelPath)

  setDefault(identifierCol -> "id", distanceFunction -> "cosine", numPartitions -> 1, numReplicas -> 0)
}

/**
  * Persists a knn model.
  *
  * @param instance the instance to persist
  *
  * @tparam TModel type of the model
  * @tparam TId type of the index item identifier
  * @tparam TVector type of the index item vector
  * @tparam TItem type of the index item
  * @tparam TDistance type of distance
  * @tparam TIndex type of the index
  */
private[knn] class KnnModelWriter[
  TModel <: KnnModelBase[TModel],
  TId: TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TDistance: TypeTag,
  TIndex <: Index[TId, TVector, TItem, TDistance]
] (instance: TModel with KnnModelOps[TModel, TId, TVector, TItem, TDistance, TIndex])
    extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val params =
      instance.extractParamMap().toSeq.toList
        // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
        .map { case ParamPair(param, value) => param.name -> mapper.readValue(param.jsonEncode(value), classOf[JValue]) }
        .toMap

    val metaData: JObject =
      ("class" -> instance.getClass.getName) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion", sc.version) ~
      ("uid", instance.uid) ~
      ("identifierType", typeDescription[TId]) ~
      ("vectorType", typeDescription[TVector]) ~
      ("partitions", instance.getNumPartitions) ~
      ("paramMap",  params)

    val metadataPath = new Path(path, "metadata").toString
    sc.parallelize(Seq(compact(metaData)), numSlices = 1).saveAsTextFile(metadataPath)

    val indicesPath = new Path(path, "indices").toString

    val modelOutputDir = instance.outputDir

    val serializableConfiguration = new SerializableConfiguration(sc.hadoopConfiguration)

    sc.range(start = 0, end = instance.getNumPartitions).foreach { partitionId =>
      val originPath = new Path(modelOutputDir, partitionId.toString)
      val originFileSystem = originPath.getFileSystem(serializableConfiguration.value)

      if (originFileSystem.exists(originPath)) {
        val destinationPath = new Path(indicesPath, partitionId.toString)
        val destinationFileSystem = destinationPath.getFileSystem(serializableConfiguration.value)
        FileUtil.copy(originFileSystem, originPath, destinationFileSystem, destinationPath, false, serializableConfiguration.value)
      }
    }
  }

  private def typeDescription[T: TypeTag] = typeOf[T] match {
    case t if t =:= typeOf[Int] => "int"
    case t if t =:= typeOf[Long] => "long"
    case t if t =:= typeOf[String] => "string"
    case t if t =:= typeOf[Array[Float]] => "float_array"
    case t if t =:= typeOf[Array[Double]] => "double_array"
    case t if t =:= typeOf[Vector] => "vector"
    case _ => "unknown"
  }
}

/**
  * Reads a knn model from persistent storage.
  *
  * @param ev classtag
  * @tparam TModel type of model
  */
private[knn] abstract class KnnModelReader[TModel <: KnnModelBase[TModel]](implicit ev: ClassTag[TModel])
  extends MLReader[TModel] {

  private implicit val format: Formats = DefaultFormats

  override def load(path: String): TModel = {

    val metadataPath = new Path(path, "metadata").toString

    val metadataStr = sc.textFile(metadataPath, 1).first()

    // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
    val metadata = mapper.readValue(metadataStr, classOf[JValue])

    val uid = (metadata \ "uid").extract[String]

    val identifierType = (metadata \ "identifierType").extract[String]
    val vectorType = (metadata \ "vectorType").extract[String]
    val partitions = (metadata \ "partitions").extract[Int]

    val paramMap = (metadata \ "paramMap").extract[JObject]

    val indicesPath = new Path(path, "indices").toString

    val model = (identifierType, vectorType) match {
      case ("int", "float_array") => createModel[Int, Array[Float], IntFloatArrayIndexItem, Float](uid, indicesPath, partitions)
      case ("int", "double_array") => createModel[Int, Array[Double], IntDoubleArrayIndexItem, Double](uid, indicesPath, partitions)
      case ("int", "vector") => createModel[Int, Vector, IntVectorIndexItem, Double](uid, indicesPath, partitions)

      case ("long", "float_array") => createModel[Long, Array[Float], LongFloatArrayIndexItem, Float](uid, indicesPath, partitions)
      case ("long", "double_array") => createModel[Long, Array[Double], LongDoubleArrayIndexItem, Double](uid, indicesPath, partitions)
      case ("long", "vector") => createModel[Long, Vector, LongVectorIndexItem, Double](uid, indicesPath, partitions)

      case ("string", "float_array") => createModel[String, Array[Float], StringFloatArrayIndexItem, Float](uid, indicesPath, partitions)
      case ("string", "double_array") => createModel[String, Array[Double], StringDoubleArrayIndexItem, Double](uid, indicesPath, partitions)
      case ("string", "vector") => createModel[String, Vector, StringVectorIndexItem, Double](uid, indicesPath, partitions)
      case _ => throw new IllegalStateException(s"Cannot create model for identifier type $identifierType and vector type $vectorType.")
    }

    paramMap.obj.foreach { case (paramName, jsonValue) =>
      val param = model.getParam(paramName)
      model.set(param, param.jsonDecode(compact(render(jsonValue))))
    }

    model
  }

  /**
    * Creates the model to be returned from fitting the data.
    *
    * @param uid identifier
    * @param outputDir directory containing the persisted indices
    * @param numPartitions number of index partitions
    *
    * @tparam TId type of the index item identifier
    * @tparam TVector type of the index item vector
    * @tparam TItem type of the index item
    * @tparam TDistance type of distance between items
    * @return model
    */
  protected def createModel[
    TId : TypeTag,
    TVector : TypeTag,
    TItem <: Item[TId, TVector] with Product : TypeTag,
    TDistance: TypeTag
  ](uid: String, outputDir: String, numPartitions: Int)
    (implicit ev: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : TModel

}

/**
 * Base class for nearest neighbor search models.
 *
 * @tparam TModel type of the model
 **/
private[knn] abstract class KnnModelBase[TModel <: KnnModelBase[TModel]] extends Model[TModel] with KnnModelParams {

  private[knn] def outputDir: String

  def getNumPartitions: Int

  /** @group setParam */
  def setQueryIdentifierCol(value: String): this.type = set(queryIdentifierCol, value)

  /** @group setParam */
  def setQueryPartitionsCol(value: String): this.type = set(queryPartitionsCol, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  /** @group setParam */
  def setSimilarityThreshold(value: Double): this.type = set(similarityThreshold, value)

  /** @group setParam */
  def setNumReplicas(value: Int): this.type = set(numReplicas, value)

  /** @group setParam */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /** @group setParam */
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

}

/**
  * Contains the core knn search logic
  *
  * @tparam TModel type of the model
  * @tparam TId type of the index item identifier
  * @tparam TVector type of the index item vector
  * @tparam TItem type of the index item
  * @tparam TDistance type of distance between items
  * @tparam TIndex type of the index
  */
private[knn] trait KnnModelOps[
  TModel <: KnnModelBase[TModel],
  TId,
  TVector,
  TItem <: Item[TId, TVector] with Product,
  TDistance,
  TIndex <: Index[TId, TVector, TItem, TDistance]
] {
  this: TModel with KnnModelParams =>

  protected def loadIndex(in: InputStream): TIndex

  protected def typedTransform(dataset: Dataset[_])
                              (implicit tId: TypeTag[TId], tVector: TypeTag[TVector], tDistance: TypeTag[TDistance], evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : DataFrame = {

    if (!isSet(queryIdentifierCol) && getExcludeSelf) {
      throw new IllegalArgumentException("QueryIdentifierCol must be defined when excludeSelf is true.")
    }

    if (isSet(queryIdentifierCol)) typedTransformWithQueryCol[TId](dataset, getQueryIdentifierCol)
    else typedTransformWithQueryCol[Long](dataset.withColumn("_query_id", monotonically_increasing_id), "_query_id").drop("_query_id")
  }

  protected def typedTransformWithQueryCol[TQueryId](dataset: Dataset[_], queryIdCol: String)
                                                    (implicit tId: TypeTag[TId], tVector: TypeTag[TVector], tDistance: TypeTag[TDistance], tQueryId: TypeTag[TQueryId], evId: ClassTag[TId], evVector: ClassTag[TVector], evQueryId: ClassTag[TQueryId], distanceNumeric: Numeric[TDistance]) : DataFrame = {
    import dataset.sparkSession.implicits._
    import distanceNumeric._

    implicit val encoder: Encoder[TQueryId] = ExpressionEncoder()
    implicit val neighborOrdering: Ordering[Neighbor[TId, TDistance]] = Ordering.by(_.distance)

    val serializableHadoopConfiguration = new SerializableConfiguration(dataset.sparkSession.sparkContext.hadoopConfiguration)

    // construct the queries to the distributed indices. when query partitions are specified we only query those partitions
    // otherwise we query all partitions
    val logicalPartitionAndQueries =
      if (isDefined(queryPartitionsCol)) dataset
        .select(
          col(getQueryPartitionsCol),
          col(queryIdCol),
          col(getFeaturesCol)
        )
        .as[(Seq[Int], TQueryId, TVector)]
        .rdd
        .flatMap { case (queryPartitions, queryId, vector) =>
          queryPartitions.map { partition => (partition, (queryId, vector)) }
        }
      else dataset
        .select(
          col(queryIdCol),
          col(getFeaturesCol)
        )
        .as[(TQueryId, TVector)]
        .rdd
        .flatMap { case (queryId, vector) =>
          Range(0, getNumPartitions).map { partition =>
            (partition, (queryId, vector))
          }
        }

    val numPartitionCopies = getNumReplicas + 1

    val physicalPartitionAndQueries = logicalPartitionAndQueries
      .map { case (partition, (queryId, vector)) =>
        val randomCopy = ThreadLocalRandom.current().nextInt(numPartitionCopies)
        val physicalPartition = (partition * numPartitionCopies) + randomCopy
        (physicalPartition, (queryId, vector))
      }
      .partitionBy(new PartitionIdPassthrough(getNumPartitions * numPartitionCopies))

    val numThreads =
      if (isSet(parallelism) && getParallelism <= 0) sys.runtime.availableProcessors
      else if (isSet(parallelism)) getParallelism
      else dataset.sparkSession.sparkContext.getConf.getInt("spark.task.cpus", defaultValue = 1)

    val neighborsOnAllQueryPartitions = physicalPartitionAndQueries
      .mapPartitions { queriesWithPartition =>

        val queries = queriesWithPartition.map(_._2)

        // load the partitioned index and execute all queries.

        val physicalPartitionId = TaskContext.getPartitionId()

        val logicalPartitionId = physicalPartitionId / numPartitionCopies
        val replica = physicalPartitionId % numPartitionCopies

        val indexPath = new Path(outputDir, logicalPartitionId.toString)

        val fileSystem = indexPath.getFileSystem(serializableHadoopConfiguration.value)

        if (!fileSystem.exists(indexPath)) Iterator.empty
        else {

          logInfo(logicalPartitionId, replica, s"started loading index from $indexPath on host ${InetAddress.getLocalHost.getHostName}")
          val index = loadIndex(fileSystem.open(indexPath))
          logInfo(logicalPartitionId, replica, s"finished loading index from $indexPath on host ${InetAddress.getLocalHost.getHostName}")

          // execute queries in parallel on multiple threads
          new Iterator[(TQueryId, Seq[Neighbor[TId, TDistance]])] {

            private[this] var first = true
            private[this] var count = 0

            private[this] val batchSize = 1000
            private[this] val queue = new LinkedBlockingQueue[(TQueryId, Seq[Neighbor[TId, TDistance]])](batchSize * numThreads)
            private[this] val executorService = new ThreadPoolExecutor(numThreads, numThreads, 60L,
              TimeUnit.SECONDS, new LinkedBlockingQueue[Runnable], new NamedThreadFactory("searcher-%d")) {
              override def afterExecute(r: Runnable, t: Throwable): Unit = {
                super.afterExecute(r, t)

                Option(t).orElse {
                  r match {
                    case t: FutureTask[_] => Try(t.get()).failed.toOption.map {
                      case e: ExecutionException => e.getCause
                      case e: InterruptedException =>
                        Thread.currentThread().interrupt()
                        e
                      case NonFatal(e) => e
                    }
                    case _ => None
                  }
                }.foreach { e =>
                  logError("Error in worker.", e)
                }
              }
            }
            executorService.allowCoreThreadTimeOut(true)

            private[this] val activeWorkers = new CountDownLatch(numThreads)
            Range(0, numThreads).map(_ => new Worker(queries, activeWorkers, batchSize)).foreach(executorService.submit)

            override def hasNext: Boolean = {
              if (!queue.isEmpty) true
              else if (queries.synchronized { queries.hasNext }) true
              else {
                // in theory all workers could have just picked up the last new work but not started processing any of it.
                if (!activeWorkers.await(2, TimeUnit.MINUTES)) {
                  throw new IllegalStateException("Workers failed to complete.")
                }
                !queue.isEmpty
              }
            }

            override def next(): (TQueryId, Seq[Neighbor[TId, TDistance]]) = {
              if (first) {
                logInfo(logicalPartitionId, replica, s"started querying on host ${InetAddress.getLocalHost.getHostName} with ${sys.runtime.availableProcessors} available processors.")
                first  = false
              }

              val value = queue.poll(1, TimeUnit.MINUTES)

              count += 1

              if (!hasNext) {
                logInfo(logicalPartitionId, replica, s"finished querying $count items on host ${InetAddress.getLocalHost.getHostName}")

                executorService.shutdown()
              }

              value
            }

            class Worker(queries: Iterator[(TQueryId, TVector)], activeWorkers: CountDownLatch, batchSize: Int) extends Runnable {

              private[this] var work = List.empty[(TQueryId, TVector)]

              private[this] val fetchSize =
                if (getExcludeSelf) getK + 1
                else getK

              @tailrec final override def run(): Unit = {

                work.foreach { case (id, vector) =>

                  val neighbors = index.findNearest(vector, fetchSize)
                    .collect { case SearchResult(item, distance)
                      if (!getExcludeSelf || item.id != id) && (getSimilarityThreshold < 0 || distance.toDouble < getSimilarityThreshold) =>
                        Neighbor[TId, TDistance](item.id, distance)
                    }

                  queue.put(id -> neighbors)
                }

                work = queries.synchronized {
                  queries.take(batchSize).toList
                }

                if (work.nonEmpty) {
                  run()
                } else {
                  activeWorkers.countDown()
                }
              }
            }
          }
        }
      }.toDS()

    // take the best k results from all partitions

    val topNeighbors = neighborsOnAllQueryPartitions
      .groupByKey { case (queryId, _) => queryId }
      .flatMapGroups { case (queryId, groups) =>
        val allNeighbors = groups.flatMap { case (_, neighbors) => neighbors}.toList
        Iterator.single(queryId -> allNeighbors.sortBy(_.distance).take(getK))
      }
      .toDF(queryIdCol, getPredictionCol)

    if (getOutputFormat == "minimal") topNeighbors
    else dataset.join(topNeighbors, Seq(queryIdCol))
  }

  protected def typedTransformSchema[T: TypeTag](schema: StructType): StructType = {
    val idDataType = typeOf[T] match {
      case t if t =:= typeOf[Int] => IntegerType
      case t if t =:= typeOf[Long] => LongType
      case _ => StringType
    }
    validateAndTransformSchema(schema, idDataType)
  }

  private def logInfo(partition: Int, replica: Int, message: String): Unit =
    logInfo(f"partition $partition%04d replica $replica%04d: $message")

}

private[knn] abstract class KnnAlgorithm[TModel <: KnnModelBase[TModel]](override val uid: String)
  extends Estimator[TModel] with KnnAlgorithmParams {

  /**
    * Type of index.
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    */
  protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] <: Index[TId, TVector, TItem, TDistance]

  /** @group setParam */
  def setIdentifierCol(value: String): this.type = set(identifierCol, value)

  /** @group setParam */
  def setQueryIdentifierCol(value: String): this.type = set(queryIdentifierCol, value)

  /** @group setParam */
  def setPartitionCol(value: String): this.type = set(partitionCol, value)

  /** @group setParam */
  def setQueryPartitionsCol(value: String): this.type = set(queryPartitionsCol, value)

  /** @group setParam */
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  /** @group setParam */
  def setSimilarityThreshold(value: Double): this.type = set(similarityThreshold, value)

  /** @group setParam */
  def setNumReplicas(value: Int): this.type = set(numReplicas, value)

  /** @group setParam */
  def setParallelism(value: Int): this.type = set(parallelism, value)

  /** @group setParam */
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  def setInitialModelPath(value: String): this.type = set(initialModelPath, value)

  override def fit(dataset: Dataset[_]): TModel = {

    val identifierType = dataset.schema(getIdentifierCol).dataType
    val vectorType = dataset.schema(getFeaturesCol).dataType

    val model = (identifierType, vectorType) match {
      case (IntegerType, ArrayType(FloatType, _)) => typedFit[Int, Array[Float], IntFloatArrayIndexItem, Float](dataset)
      case (IntegerType, ArrayType(DoubleType, _)) => typedFit[Int, Array[Double], IntDoubleArrayIndexItem, Double](dataset)
      case (IntegerType, VectorType) => typedFit[Int, Vector, IntVectorIndexItem, Double](dataset)
      case (LongType, ArrayType(FloatType, _)) => typedFit[Long, Array[Float], LongFloatArrayIndexItem, Float](dataset)
      case (LongType, ArrayType(DoubleType, _)) => typedFit[Long, Array[Double], LongDoubleArrayIndexItem, Double](dataset)
      case (LongType, VectorType)  => typedFit[Long, Vector, LongVectorIndexItem, Double](dataset)
      case (StringType, ArrayType(FloatType, _)) => typedFit[String, Array[Float], StringFloatArrayIndexItem, Float](dataset)
      case (StringType, ArrayType(DoubleType, _)) => typedFit[String, Array[Double], StringDoubleArrayIndexItem, Double](dataset)
      case (StringType, VectorType) => typedFit[String, Vector, StringVectorIndexItem, Double](dataset)
      case _ =>
        throw new IllegalArgumentException(s"Cannot create index for items with identifier of type " +
        s"${identifierType.simpleString} and vector of type ${vectorType.simpleString}. " +
        s"Supported identifiers are string, int, long and string. Supported vectors are array<float>, array<double> and vector ")
    }

    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, schema(getIdentifierCol).dataType)

  override def copy(extra: ParamMap): Estimator[TModel] = defaultCopy(extra)

  /**
    * Create the index used to do the nearest neighbor search.
    *
    * @param dimensions dimensionality of the items stored in the index
    * @param maxItemCount maximum number of items the index can hold
    * @param distanceFunction the distance function
    * @param distanceOrdering the distance ordering
    * @param idSerializer invoked for serializing ids when saving the index
    * @param itemSerializer invoked for serializing items when saving items
    *
    * @tparam TId type of the index item identifier
    * @tparam TVector type of the index item vector
    * @tparam TItem type of the index item
    * @tparam TDistance type of distance between items
    * @return create an index
    */
  protected def createIndex[
    TId,
    TVector,
    TItem <: Item[TId, TVector] with Product,
    TDistance
  ](dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, TDistance])
    (implicit distanceOrdering: Ordering[TDistance], idSerializer: ObjectSerializer[TId], itemSerializer: ObjectSerializer[TItem])
      : TIndex[TId, TVector, TItem, TDistance]

  /**
   * Load an index
   *
   * @param inputStream InputStream to restore the index from
   * @param minCapacity loaded index needs to have space for at least this man additional items
   *
   * @tparam TId       type of the index item identifier
   * @tparam TVector   type of the index item vector
   * @tparam TItem     type of the index item
   * @tparam TDistance type of distance between items
   * @return create an index
   */
  protected def loadIndex[TId, TVector, TItem <: Item[TId, TVector] with Product, TDistance](inputStream: InputStream,
                                                                                             minCapacity: Int)
    : TIndex[TId, TVector, TItem, TDistance]

  /**
    * Creates the model to be returned from fitting the data.
    *
    * @param uid identifier
    * @param outputDir directory containing the persisted indices
    * @param numPartitions number of index partitions
    *
    * @tparam TId type of the index item identifier
    * @tparam TVector type of the index item vector
    * @tparam TItem type of the index item
    * @tparam TDistance type of distance between items
    * @return model
    */
  protected def createModel[
    TId : TypeTag,
    TVector : TypeTag,
    TItem <: Item[TId, TVector] with Product : TypeTag,
    TDistance: TypeTag
  ](uid: String, outputDir: String, numPartitions: Int)
    (implicit ev: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance])
      : TModel

  private def typedFit[
    TId : TypeTag,
    TVector : TypeTag,
    TItem <: Item[TId, TVector] with Product : TypeTag,
    TDistance: TypeTag
  ](dataset: Dataset[_])
    (implicit ev: ClassTag[TId], evVector: ClassTag[TVector], evItem: ClassTag[TItem], distanceNumeric: Numeric[TDistance], distanceFunctionFactory: String => DistanceFunction[TVector, TDistance], idSerializer: ObjectSerializer[TId], itemSerializer: ObjectSerializer[TItem])
      : TModel = {

    val sc = dataset.sparkSession
    val sparkContext = sc.sparkContext

    val serializableHadoopConfiguration = new SerializableConfiguration(sparkContext.hadoopConfiguration)

    import sc.implicits._

    val cacheFolder = sparkContext.getConf.get(key = "spark.hnswlib.settings.index.cache_folder", defaultValue = "/tmp")

    val outputDir = new Path(cacheFolder,s"${uid}_${System.currentTimeMillis()}").toString

    sparkContext.addSparkListener(new CleanupListener(outputDir, serializableHadoopConfiguration))

    // read the id and vector from the input dataset and and repartition them over numPartitions amount of partitions.
    // if the data is pre-partitioned by the user repartition the input data by the user defined partition key, use a
    // hash of the item id otherwise.
    val partitionedIndexItems = {
      if (isDefined(partitionCol)) dataset
        .select(
          col(getPartitionCol).as("partition"),
          struct(col(getIdentifierCol).as("id"), col(getFeaturesCol).as("vector"))
        )
        .as[(Int, TItem)]
        .rdd
        .partitionBy(new PartitionIdPassthrough(getNumPartitions))
        .values
        .toDS
      else dataset
        .select(
          col(getIdentifierCol).as("id"),
          col(getFeaturesCol).as("vector"))
        .as[TItem]
        .repartition(getNumPartitions, $"id")
    }

    // On each partition collect all the items into memory and construct the HNSW indices.
    // Save these indices to the hadoop filesystem

    val numThreads =
      if (isSet(parallelism) && getParallelism <= 0) sys.runtime.availableProcessors
      else if (isSet(parallelism)) getParallelism
      else dataset.sparkSession.sparkContext.getConf.getInt("spark.task.cpus", defaultValue = 1)

    val initialModelOutputDir =
      if (isSet(initialModelPath)) Some(new Path(getInitialModelPath, "indices").toString)
      else None

    val serializableConfiguration = new SerializableConfiguration(sparkContext.hadoopConfiguration)

    partitionedIndexItems
      .foreachPartition { it: Iterator[TItem] =>
        if (it.hasNext) {
          val partitionId = TaskContext.getPartitionId()

          val items = it.toSeq

          logInfo(partitionId,s"started indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val existingIndexOption = initialModelOutputDir
            .flatMap { dir =>
              val indexPath = new Path(dir, partitionId.toString)
              val fs = indexPath.getFileSystem(serializableConfiguration.value)

              if (fs.exists(indexPath)) Some {
                val inputStream = fs.open(indexPath)
                loadIndex[TId, TVector, TItem, TDistance](inputStream, items.size)
              } else {
                logInfo(partitionId, s"File $indexPath not found.")
                None
              }
            }

          val index = existingIndexOption
            .getOrElse(createIndex[TId, TVector, TItem, TDistance](items.head.dimensions, items.size, distanceFunctionFactory(getDistanceFunction)))

          index.addAll(items, progressUpdateInterval = 5000, listener = (workDone, max) => logDebug(f"partition $partitionId%04d: Indexed $workDone of $max items"), numThreads = numThreads)

          logInfo(partitionId, s"finished indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val path = new Path(outputDir, partitionId.toString)
          val fileSystem = path.getFileSystem(serializableHadoopConfiguration.value)

          val outputStream = fileSystem.create(path)

          logInfo(partitionId, s"started saving index to $path on host ${InetAddress.getLocalHost.getHostName}")

          index.save(outputStream)

          logInfo(partitionId, s"finished saving index to $path on host ${InetAddress.getLocalHost.getHostName}")
        }
      }

    createModel[TId, TVector, TItem, TDistance](uid, outputDir, getNumPartitions)
  }

  private def logInfo(partition: Int, message: String): Unit = logInfo(f"partition $partition%04d: $message")

  implicit private def floatArrayDistanceFunction(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case value => userDistanceFunction(value)
  }

  implicit private def doubleArrayDistanceFunction(name: String): DistanceFunction[Array[Double], Double] = name match {
    case "bray-curtis" => doubleBrayCurtisDistance
    case "canberra" => doubleCanberraDistance
    case "correlation" => doubleCorrelationDistance
    case "cosine" => doubleCosineDistance
    case "euclidean" => doubleEuclideanDistance
    case "inner-product" => doubleInnerProduct
    case "manhattan" => doubleManhattanDistance
    case value => userDistanceFunction(value)
  }

  implicit private def vectorDistanceFunction(name: String): DistanceFunction[Vector, Double] = name match {
    case "bray-curtis" => VectorDistanceFunctions.brayCurtisDistance
    case "canberra" => VectorDistanceFunctions.canberraDistance
    case "correlation" => VectorDistanceFunctions.correlationDistance
    case "cosine" => VectorDistanceFunctions.cosineDistance
    case "euclidean" => VectorDistanceFunctions.euclideanDistance
    case "inner-product" => VectorDistanceFunctions.innerProduct
    case "manhattan" => VectorDistanceFunctions.manhattanDistance
    case value => userDistanceFunction(value)
  }

  private def userDistanceFunction[TVector, TDistance](name: String): DistanceFunction[TVector, TDistance] =
    Try(Class.forName(name).getDeclaredConstructor().newInstance())
      .toOption
      .collect { case f: DistanceFunction[TVector @unchecked, TDistance @unchecked] => f }
      .getOrElse(throw new IllegalArgumentException(s"$name is not a valid distance functions."))
}

private[knn] class CleanupListener(dir: String, serializableConfiguration: SerializableConfiguration) extends SparkListener with Logging {
  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {

    val path = new Path(dir)
    val fileSystem = path.getFileSystem(serializableConfiguration.value)

    logInfo(s"Deleting files below $dir")
    fileSystem.delete(path, true)
  }
}

/**
  * Partitioner that uses precomputed partitions
  *
  * @param numPartitions number of partitions
  */
private[knn] class PartitionIdPassthrough(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}


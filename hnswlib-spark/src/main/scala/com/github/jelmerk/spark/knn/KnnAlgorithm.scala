package com.github.jelmerk.spark.knn

import java.io.InputStream
import java.net.InetAddress
import java.util.concurrent.ThreadLocalRandom

import com.github.jelmerk.knn.ObjectSerializer

import scala.language.{higherKinds, implicitConversions}
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
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
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.spark.linalg.functions.VectorDistanceFunctions
import com.github.jelmerk.spark.util.BoundedPriorityQueue


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
    * Param for the column name for the row identifier.
    * Default: "id"
    *
    * @group param
    */
  val identifierCol = new Param[String](this, "identifierCol", "column name for the row identifier")

  /** @group getParam */
  def getIdentifierCol: String = $(identifierCol)

  /**
    * Param for the column name for the query identifier.
    *
    * @group param
    */
  val queryIdentifierCol = new Param[String](this, "queryIdentifierCol", "column name for the query identifier")

  /** @group getParam */
  def getQueryIdentifierCol: String = $(queryIdentifierCol)

  /**
    * Param for number of neighbors to find (> 0).
    * Default: 5
    *
    * @group param
    */
  val k = new IntParam(this, "k", "number of neighbors to find", ParamValidators.gt(0))

  /** @group getParam */
  def getK: Int = $(k)

  /**
    * Param that indicates whether to not return the a candidate when it's identifier equals the query identifier
    * Default: false
    *
    * @group param
    */
  val excludeSelf = new BooleanParam(this, "excludeSelf", "whether to include the row identifier as a candidate neighbor")

  /** @group getParam */
  def getExcludeSelf: Boolean = $(excludeSelf)

  /**
    * Param for the threshold value for inclusion. -1 indicates no threshold
    * Default: -1
    *
    * @group param
    */
  val similarityThreshold = new DoubleParam(this, "similarityThreshold", "do not return neighbors further away than this distance")

  /** @group getParam */
  def getSimilarityThreshold: Double = $(similarityThreshold)

  /**
    * Param that specifies the number of index replicas to create when querying the index. More replicas means you can
    * execute more queries in parallel at the expense of increased resource usage.
    * Default: 0
    *
    * @group param
    */
  val numReplicas = new IntParam(this, "numReplicas", "number of index replicas to create when querying")

  /** @group getParam */
  def getNumReplicas: Int = $(numReplicas)

  /**
    * Param for the output format to produce. One of "full", "minimal" Setting this to minimal is more efficient
    * when all you need is the identifier with its neighbors
    *
    * Default: "full"
    *
    * @group param
    */
  val outputFormat = new Param[String](this, "outputFormat", "output format to produce")

  /** @group getParam */
  def getOutputFormat: String = $(outputFormat)

  setDefault(k -> 5, predictionCol -> "prediction", identifierCol -> "id", featuresCol -> "features",
    excludeSelf -> false, similarityThreshold -> -1, outputFormat -> "full")

  protected def validateAndTransformSchema(schema: StructType): StructType = {

    val identifierColSchema = schema(getIdentifierCol)

    val distanceType = schema(getFeaturesCol).dataType match {
      case ArrayType(FloatType, _) => FloatType
      case _ => DoubleType
    }

    val neighborsField = StructField(getPredictionCol, ArrayType(StructType(Seq(StructField("neighbor", identifierColSchema.dataType, identifierColSchema.nullable), StructField("distance", distanceType)))))

    getOutputFormat match {
      case "minimal" => StructType(Array(identifierColSchema, neighborsField))

      case _ =>
        if (schema.fieldNames.contains(getPredictionCol)) {
          throw new IllegalArgumentException(s"Output column $getPredictionCol already exists.")
        }

        StructType(schema.fields :+ neighborsField)
    }
  }
}

/**
  * Params for knn algorithms.
  */
private[knn] trait KnnAlgorithmParams extends KnnModelParams {

  /**
    * Number of partitions (default: 1)
    */
  val numPartitions = new IntParam(this, "numPartitions",
    "number of partitions", ParamValidators.gt(0))

  /** @group getParam */
  def getNumPartitions: Int = $(numPartitions)

  /**
    * Param for the distance function to use. One of "bray-curtis", "canberra",  "cosine", "correlation", "euclidean",
    * "inner-product", "manhattan" or the fully qualified classname of a distance function
    * Default: "cosine"
    *
    * @group param
    */
  val distanceFunction = new Param[String](this, "distanceFunction", "column names for returned neighbors")

  /** @group getParam */
  def getDistanceFunction: String = $(distanceFunction)


  setDefault(distanceFunction -> "cosine", numPartitions -> 1, numReplicas -> 0)
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
  TModel <: Model[TModel],
  TId: TypeTag,
  TVector : TypeTag,
  TItem <: Item[TId, TVector] with Product : TypeTag,
  TDistance: TypeTag,
  TIndex <: Index[TId, TVector, TItem, TDistance]
] (instance: TModel with KnnModelOps[TModel, TId, TVector, TItem, TDistance, TIndex])
    extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val metaData = JObject(
      JField("class", JString(instance.getClass.getName)),
      JField("timestamp", JInt(System.currentTimeMillis())),
      JField("sparkVersion", JString(sc.version)),
      JField("uid", JString(instance.uid)),
      JField("identifierType", JString(typeDescription[TId])),
      JField("vectorType", JString(typeDescription[TVector])),
      JField("partitions", JInt(instance.numPartitions)),
      JField("paramMap", JObject(
        instance.extractParamMap().toSeq.toList.map { case ParamPair(param, value) =>
          // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
          JField(param.name, mapper.readValue(param.jsonEncode(value), classOf[JValue]))
        }
      ))
    )

    val metadataPath = new Path(path, "metadata").toString
    sc.parallelize(Seq(compact(metaData)), numSlices = 1).saveAsTextFile(metadataPath)

    val indicesPath = new Path(path, "indices").toString

    val modelOutputDir = instance.outputDir

    sc.range(start = 0, end = instance.numPartitions).foreach { partitionId =>
      val hadoopConfiguration = new Configuration() // should come from sc.hadoopConfiguration but it's not serializable any options ? ..
      val fileSystem = FileSystem.get(hadoopConfiguration)

      val originPath = new Path(modelOutputDir, partitionId.toString)

      if (fileSystem.exists(originPath)) {
        val destinationPath = new Path(indicesPath, partitionId.toString)
        FileUtil.copy(fileSystem, originPath, fileSystem, destinationPath, false, hadoopConfiguration)
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
  * Reads a knn model from persistent stotage.
  *
  * @param ev classtag
  * @tparam TModel type of model
  */
private[knn] abstract class KnnModelReader[TModel <: Model[TModel]](implicit ev: ClassTag[TModel])
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
private[knn] abstract class KnnModelBase[TModel <: Model[TModel]] extends Model[TModel] with KnnModelParams {

  /** @group setParam */
  def setQueryIdentifierCol(value: String): this.type = set(queryIdentifierCol, value)

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
  TModel <: Model[TModel],
  TId,
  TVector,
  TItem <: Item[TId, TVector] with Product,
  TDistance,
  TIndex <: Index[TId, TVector, TItem, TDistance]
] {
  this: TModel with KnnModelParams =>

  private[knn] def outputDir: String

  private[knn] def numPartitions: Int

  protected def loadIndex(in: InputStream): TIndex

  protected def typedTransform(dataset: Dataset[_])
                              (implicit tId: TypeTag[TId], tVector: TypeTag[TVector], tDistance: TypeTag[TDistance], evId: ClassTag[TId], evVector: ClassTag[TVector], distanceNumeric: Numeric[TDistance]) : DataFrame =
    if (isSet(queryIdentifierCol)) typedTransformWithQueryCol[TId](dataset, getQueryIdentifierCol)
    else typedTransformWithQueryCol[Long](dataset.withColumn("_query_id", monotonically_increasing_id), "_query_id").drop("_query_id")

  protected def typedTransformWithQueryCol[TQueryId](dataset: Dataset[_], queryIdCol: String)
                                                    (implicit tId: TypeTag[TId], tVector: TypeTag[TVector], tDistance: TypeTag[TDistance], tQueryId: TypeTag[TQueryId], evId: ClassTag[TId], evVector: ClassTag[TVector], evQueryId: ClassTag[TQueryId], distanceNumeric: Numeric[TDistance]) : DataFrame = {
    import dataset.sparkSession.implicits._
    import distanceNumeric._

    implicit val neighborOrdering: Ordering[Neighbor[TId, TDistance]] = Ordering.by(_.distance)

    // replicate the indices if numReplicas > 0 this means we can parallelize the querying

    val numPartitionCopies = getNumReplicas + 1

    // read the items and duplicate the rows in the query dataset with the number of partitions, assign a different partition to each copy

    val queryDs = for {
      queryIdAndVector <- dataset.select(col(queryIdCol), col(getFeaturesCol)).as[(TQueryId, TVector)]
      partition <- Range(0, numPartitions)
    } yield {
      val randomCopy = ThreadLocalRandom.current().nextInt(numPartitionCopies)
      val physicalPartition = (partition * numPartitionCopies) + randomCopy
      physicalPartition -> queryIdAndVector
    }

    // query all the indices

    val neighborsOnAllShards = queryDs.rdd
      .partitionBy(new PartitionIdPassthrough(numPartitions * numPartitionCopies))
      .mapPartitions(it => it.map { case (_, queryIdAndVector) => queryIdAndVector}, preservesPartitioning = true)
      .mapPartitions { queries =>
        val physicalPartitionId = TaskContext.getPartitionId()

        val partition = physicalPartitionId / numPartitionCopies
        val replica = physicalPartitionId % numPartitionCopies

        val hadoopConfiguration = new Configuration() // should come from sc.hadoopConfiguration but it's not serializable any options ? ..
        val fileSystem = FileSystem.get(hadoopConfiguration)

        val indexPath = new Path(outputDir, partition.toString)

        if (!fileSystem.exists(indexPath)) Iterator.empty
        else {
          logInfo(partition, replica, s"started loading index from $indexPath on host ${InetAddress.getLocalHost.getHostName}")
          val index = loadIndex(fileSystem.open(indexPath))
          logInfo(partition, replica, s"finished loading index from $indexPath on host ${InetAddress.getLocalHost.getHostName}")

          for {
            batch <- new LoggingIterator(partition, replica, queries.grouped(20480))
            queryIdAndCandidates <- batch.par.map { case (id, vector) =>
              val fetchSize =
                if (getExcludeSelf) getK + 1
                else getK

              val neighbors = index.findNearest(vector, fetchSize)
                .collect { case SearchResult(item, distance)
                  if (!getExcludeSelf || item.id != id) && (getSimilarityThreshold < 0 || distance.toDouble < getSimilarityThreshold) =>
                    Neighbor[TId, TDistance](item.id, distance)
                }

              val queue = new BoundedPriorityQueue[Neighbor[TId, TDistance]](getK)(neighborOrdering.reverse)
              queue ++= neighbors

              (id, queue)
            }
          } yield queryIdAndCandidates
        }
      }

    // reduce the top k neighbors on each shard to the top k neighbors over all shards, holding on to only the best matches

    val topNeighbors = neighborsOnAllShards
      .reduceByKey { case (neighborsA, neighborsB) =>
        neighborsA ++= neighborsB
        neighborsA
      }
      .mapValues(_.toArray.sorted(neighborOrdering))
      .toDF(queryIdCol, getPredictionCol)

    if (getOutputFormat == "minimal") topNeighbors
    else dataset.join(topNeighbors, Seq(queryIdCol))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  private def logInfo(partition: Int, replica: Int, message: String): Unit =
    logInfo(f"partition $partition%04d replica $replica%04d: $message")

  private class LoggingIterator[T](partition: Int, replica: Int, delegate: Iterator[T]) extends Iterator[T] {

    private[this] var count = 0
    private[this] var first = true

    override def hasNext: Boolean = delegate.hasNext

    override def next(): T = {
      if (first) {
        logInfo(partition, replica, s"started querying on host ${InetAddress.getLocalHost.getHostName} with ${sys.runtime.availableProcessors} available processors.")
        first  = false
      }

      val value = delegate.next()

      count += 1

      if (!hasNext) {
        logInfo(partition, replica, s"finished querying $count items on host ${InetAddress.getLocalHost.getHostName}")
      }

      value
    }
  }
}

private[knn] abstract class KnnAlgorithm[TModel <: Model[TModel]](override val uid: String)
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
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  override def fit(dataset: Dataset[_]): TModel = {

    val identifierType = dataset.schema(getIdentifierCol).dataType
    val vectorType = dataset.schema(getFeaturesCol).dataType

    val model = (identifierType, vectorType) match {
      case (IntegerType, ArrayType(FloatType, _)) => typedFit[Int, Array[Float], IntFloatArrayIndexItem, Float](dataset)
      case (IntegerType, ArrayType(DoubleType, _)) => typedFit[Int, Array[Double], IntDoubleArrayIndexItem, Double](dataset)
      case (IntegerType, t) if t.typeName == "vector" => typedFit[Int, Vector, IntVectorIndexItem, Double](dataset)
      case (LongType, ArrayType(FloatType, _)) => typedFit[Long, Array[Float], LongFloatArrayIndexItem, Float](dataset)
      case (LongType, ArrayType(DoubleType, _)) => typedFit[Long, Array[Double], LongDoubleArrayIndexItem, Double](dataset)
      case (LongType, t) if t.typeName == "vector" => typedFit[Long, Vector, LongVectorIndexItem, Double](dataset)
      case (StringType, ArrayType(FloatType, _)) => typedFit[String, Array[Float], StringFloatArrayIndexItem, Float](dataset)
      case (StringType, ArrayType(DoubleType, _)) => typedFit[String, Array[Double], StringDoubleArrayIndexItem, Double](dataset)
      case (StringType, t) if t.typeName == "vector" => typedFit[String, Vector, StringVectorIndexItem, Double](dataset)
      case _ => throw new IllegalArgumentException(s"Cannot create index for items with identifier of type " +
        s"${identifierType.simpleString} and vector of type ${vectorType.simpleString}. " +
        s"Supported identifiers are string, int, long and string. Supported vectors are array<float>, array<double> and vector ")
    }

    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

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

    import sc.implicits._

    val outputDir = sparkContext.getCheckpointDir
      .map (path => new Path(path,s"${uid}_${System.currentTimeMillis()}").toString)
      .getOrElse(throw new IllegalStateException("Please define checkpoint dir with spark.sparkContext.setCheckpointDir"))

    // spark does not like huge dataframes so instead of putting the index in an rdd we write it to disk and reload it when doing the search
    // this does raise the question of when to delete these temporary files. We do this after the application ends

    sparkContext.addSparkListener(new CleanupListener(outputDir))

    val items = dataset.select(col(getIdentifierCol).as("id"), col(getFeaturesCol).as("vector"))
      .as[TItem]

    // read the id and vector from the input dataset and and repartition them over numPartitions amount of partitions.
    // Transform vectors or double arrays into float arrays for performance reasons.

    val partitionedIndexItems = items
      .repartition(getNumPartitions, $"id")

    // On each partition collect all the items into memory and construct the HNSW indices.
    // The result is a rdd that has a single row per partition containing the index

    partitionedIndexItems
      .foreachPartition { it =>
        if (it.hasNext) {
          val partitionId = TaskContext.getPartitionId()

          val items = it.toSeq

          logInfo(partitionId,f"started indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val index = createIndex[TId, TVector, TItem, TDistance](items.head.dimensions, items.size, distanceFunctionFactory(getDistanceFunction))
          index.addAll(items, progressUpdateInterval = 5000, listener = (workDone, max) => logDebug(f"partition $partitionId%04d: Indexed $workDone of $max items"))

          logInfo(partitionId, f"finished indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val hadoopConfiguration = new Configuration() // should come from sc.hadoopConfiguration but it's not serializable any options ? ..
          val fileSystem = FileSystem.get(hadoopConfiguration)

          val path = new Path(outputDir, partitionId.toString)

          val outputStream = fileSystem.create(path)

          logInfo(partitionId, f"started saving index to $path on host ${InetAddress.getLocalHost.getHostName}")

          index.save(outputStream)

          logInfo(partitionId, f"finished saving index to $path on host ${InetAddress.getLocalHost.getHostName}")
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

private[knn] class CleanupListener(dir: String) extends SparkListener with Logging {
  override def onApplicationEnd(applicationEnd: SparkListenerApplicationEnd): Unit = {
    val hadoopConfiguration = new Configuration() // should come from sc.hadoopConfiguration but it's not serializable any options ? ..
    val fileSystem = FileSystem.get(hadoopConfiguration)

    logInfo(s"Deleting files below $dir")
    fileSystem.delete(new Path(dir), true)
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


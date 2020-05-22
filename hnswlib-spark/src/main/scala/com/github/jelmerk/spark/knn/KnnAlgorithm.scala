package com.github.jelmerk.spark.knn

import java.net.InetAddress

import scala.language.{higherKinds, implicitConversions}
import scala.math.abs
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.Partitioner
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods._
import org.json4s._
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.spark.linalg.functions.VectorDistanceFunctions
import com.github.jelmerk.spark.util.{BoundedPriorityQueue, PartitionedRdd, UnsplittableSequenceFileInputFormat, Utils}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}

private[knn] case class ArrayIndexItem[TId, TComponent](id: TId, vector: Array[TComponent]) extends Item[TId, Array[TComponent]] {
  override def dimensions: Int = vector.length
}

private[knn] case class VectorIndexItem[TId](id: TId, vector: Vector) extends Item[TId, Vector] {
  override def dimensions: Int = vector.size
}

private[knn] case class Query[TId, TVector](id: TId, vector: TVector)

/**
  * Neighbor of an item
  *
  * @param neighbor identifies the neighbor
  * @param distance distance to the item
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
    */
  val similarityThreshold = new DoubleParam(this, "similarityThreshold", "do not return neighbors further away than this distance")

  /** @group getParam */
  def getSimilarityThreshold: Double = $(similarityThreshold)

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

  /**
    * Param for StorageLevel for the indices. Pass in a string representation of
    * `StorageLevel`.
    * Default: "MEMORY_ONLY".
    *
    * @group expertParam
    */
  val storageLevel = new Param[String](this, "storageLevel", "StorageLevel for the indices")

  /** @group expertGetParam */
  def getStorageLevel: String = $(storageLevel)


  setDefault(distanceFunction -> "cosine", numPartitions -> 1, storageLevel -> "MEMORY_ONLY")
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

    val identifierType = typeOf[TId] match {
      case t if t =:= typeOf[Int] => "int"
      case t if t =:= typeOf[Long] => "long"
      case _ => "string"
    }

    val vectorType = typeOf[TVector] match {
      case t if t =:= typeOf[Array[Float]] => "float_array"
      case t if t =:= typeOf[Array[Double]] => "double_array"
      case _ => "vector"
    }

    val metaData = JObject(
      JField("class", JString(instance.getClass.getName)),
      JField("timestamp", JInt(System.currentTimeMillis())),
      JField("sparkVersion", JString(sc.version)),
      JField("uid", JString(instance.uid)),
      JField("identifierType", JString(identifierType)),
      JField("vectorType", JString(vectorType)),
      JField("paramMap", JObject(
        instance.extractParamMap().toSeq.toList.map { case ParamPair(param, value) =>
          // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
          JField(param.name, mapper.readValue(param.jsonEncode(value), classOf[JValue]))
        }
      ))
    )

    val metadataPath = new Path(path, "metadata").toString
    sc.parallelize(Seq(compact(metaData)), 1).saveAsTextFile(metadataPath)

    val indicesPath = new Path(path, "indices").toString

    instance.indices.saveAsObjectFile(indicesPath)
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

  /**
    * Type of index.
    *
    * @tparam TId Type of the external identifier of an item
    * @tparam TVector Type of the vector to perform distance calculation on
    * @tparam TItem Type of items stored in the index
    * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
    */
  protected type TIndex[TId, TVector, TItem <: Item[TId, TVector], TDistance] <: Index[TId, TVector, TItem, TDistance]

  override def load(path: String): TModel = {

    val metadataPath = new Path(path, "metadata").toString

    val metadataStr = sc.textFile(metadataPath, 1).first()

    // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
    val metadata = mapper.readValue(metadataStr, classOf[JValue])

    val uid = (metadata \ "uid").extract[String]

    val identifierType = (metadata \ "identifierType").extract[String]
    val vectorType = (metadata \ "vectorType").extract[String]

    val paramMap = (metadata \ "paramMap").extract[JObject]

    val indicesPath = new Path(path, "indices").toString

    val model = (identifierType, vectorType) match {
      case ("int", "float_array") => loadModel[Int, Array[Float], ArrayIndexItem[Int, Float], Float](uid, indicesPath)
      case ("int", "double_array") => loadModel[Int, Array[Double], ArrayIndexItem[Int, Double], Double](uid, indicesPath)
      case ("int", "vector") => loadModel[Int, Vector, VectorIndexItem[Int], Double](uid, indicesPath)

      case ("long", "float_array") => loadModel[Long, Array[Float], ArrayIndexItem[Long, Float], Float](uid, indicesPath)
      case ("long", "double_array") => loadModel[Long, Array[Double], ArrayIndexItem[Long, Double], Double](uid, indicesPath)
      case ("long", "vector") => loadModel[Long, Vector, VectorIndexItem[Long], Double](uid, indicesPath)

      case ("string", "float_array") => loadModel[String, Array[Float], ArrayIndexItem[String, Float], Float](uid, indicesPath)
      case ("string", "double_array") => loadModel[String, Array[Double], ArrayIndexItem[String, Double], Double](uid, indicesPath)
      case ("string", "vector") => loadModel[String, Vector, VectorIndexItem[String], Double](uid, indicesPath)
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
    * @param indices rdd that holds the partitioned indices that are used to do the search
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
  ](uid: String, indices: RDD[(Int, TIndex[TId, TVector, TItem, TDistance])])(implicit ev: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance]) : TModel

  private def loadModel[
    TId : TypeTag,
    TVector : TypeTag,
    TItem <: Item[TId, TVector] with Product : TypeTag,
    TDistance : TypeTag
  ](uid: String, indicesPath: String)(implicit ev: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance]): TModel = {
    val serializedIndexRdd = sc.hadoopFile(indicesPath, classOf[UnsplittableSequenceFileInputFormat[NullWritable, BytesWritable]], classOf[NullWritable], classOf[BytesWritable])
    val indices = serializedIndexRdd.flatMap { case (_, value) => Utils.deserialize[Array[(Int, TIndex[TId, TVector, TItem, TDistance])]](value.getBytes) }
    val partitionedIndices = new PartitionedRdd(indices, Some(new PartitionIdPassthrough(indices.getNumPartitions)))
    createModel(uid, partitionedIndices)
  }

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


  private[knn] def indices: RDD[(Int, TIndex)]

  protected def typedTransform(dataset: Dataset[_])
                              (implicit ev1: TypeTag[TId], ev2: TypeTag[TVector], ev3: TypeTag[TDistance], evId: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance]) : DataFrame =
    if (isSet(queryIdentifierCol)) typedTransformWithQueryCol[TId](dataset, getQueryIdentifierCol)
    else typedTransformWithQueryCol[Long](dataset.withColumn("_query_id", monotonically_increasing_id), "_query_id").drop("_query_id")

  protected def typedTransformWithQueryCol[TQueryId](dataset: Dataset[_], queryIdCol: String)
                                                    (implicit ev1: TypeTag[TId], ev2: TypeTag[TVector], ev3: TypeTag[TDistance], ev4: TypeTag[TQueryId], evId: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], evQueryId: ClassTag[TQueryId], distanceOrdering: Ordering[TDistance]) : DataFrame = {
    import dataset.sparkSession.implicits._

    implicit val neighborOrdering: Ordering[Neighbor[TId, TDistance]] = Ordering.by(_.distance)

    // this works because distance is always either float or double right now this is not very nice and I should probably come up with something more robust
    val threshold = getSimilarityThreshold.asInstanceOf[TDistance]

    // read the items and duplicate the rows in the query dataset with the number of partitions, assign a different partition to each copy
    val queryRdd = dataset.select(
        col(queryIdCol),
        col(getFeaturesCol)
      )
      .withColumn("partition", explode(array(0 until indices.getNumPartitions map lit: _*)))
      .as[(TQueryId, TVector, Int)]
      .rdd
      .map { case (id, vector, partition) => (partition, (null.asInstanceOf[TIndex], id, vector)) }
      .partitionBy(indices.partitioner.get)

    // combine the indices rdd and query rdds into a single rdd and make sure the first row of the unioned rdd is our index

    val unioned = indices
      .mapPartitions(_.map { case (partition, index) => (partition, (index, null.asInstanceOf[TQueryId], null.asInstanceOf[TVector])) }, preservesPartitioning = true)
      .union(queryRdd)

    // map over all the rows in the partition, hold on on to the index stored in the first row and
    // use it to find the nearest neighbors of the remaining rows

    val neighborsOnAllShards = unioned.mapPartitions { it =>
      if (it.hasNext) {
        val (partition, (index, _, _)) = it.next()

        if (index == null) {
          logInfo(f"partition $partition%04d: No index on partition, not querying anything.")
          Iterator.empty
        } else {
          transformIndex(index)

          new LoggingIterator(partition,
            it.grouped(20480).flatMap { grouped =>

              // use scala's parallel collections to speed up querying

              grouped.par.map { case (_, (_, id, vector)) =>

                val fetchSize =
                  if (getExcludeSelf) getK + 1
                  else getK

                val neighbors = index.findNearest(vector, fetchSize)
                  .collect { case SearchResult(item, distance)
                    if (!getExcludeSelf || item.id != id) && (getSimilarityThreshold < 0 || distanceOrdering.lt(distance, threshold)) =>
                      Neighbor[TId, TDistance](item.id, distance)
                  }

                val queue = new BoundedPriorityQueue[Neighbor[TId, TDistance]](getK)(neighborOrdering.reverse)
                queue ++= neighbors

                id -> queue
              }
            }
          )
        }
      } else Iterator.empty
    }

    // reduce the top k neighbors on each shard to the top k neighbors over all shards, holding on to only the best matches

    val topNeighbors: RDD[(TQueryId, Array[Neighbor[TId, TDistance]])] = neighborsOnAllShards
      .reduceByKey { case (neighborsA, neighborsB) =>
        neighborsA ++= neighborsB
        neighborsA
      }
      .mapValues(_.toArray.sorted(neighborOrdering))

    // transform the rdd into our output dataframe

    val transformed = topNeighbors
        .toDF(queryIdCol, getPredictionCol)

    if (getOutputFormat == "minimal") transformed
    else dataset.join(transformed, Seq(queryIdCol))
  }

  /**
    * Subclasses can implement this class in order to transform the index before querying.
    *
    * @param index the index to transform
    */
  private[knn] def transformIndex(index: TIndex): Unit = ()

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  private class LoggingIterator[T](partition: Int, delegate: Iterator[T]) extends Iterator[T] {

    private[this] var count = 0
    private[this] var first = true

    override def hasNext: Boolean = delegate.hasNext

    override def next(): T = {
      if (first) {
        logInfo(f"partition $partition%04d: started querying on host ${InetAddress.getLocalHost.getHostName} with ${sys.runtime.availableProcessors} available processors.")
        first  = false
      }

      val value = delegate.next()

      count += 1

      if (!hasNext) {
        logInfo(f"partition $partition%04d: finished querying $count items on host ${InetAddress.getLocalHost.getHostName}")
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
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  /** @group setParam */
  def setStorageLevel(value: String): this.type = set(storageLevel, value)

  override def fit(dataset: Dataset[_]): TModel = {

    val identifierType = dataset.schema(getIdentifierCol).dataType
    val vectorType = dataset.schema(getFeaturesCol).dataType

    val model = (identifierType, vectorType) match {
      case (IntegerType, ArrayType(FloatType, _)) => typedFit[Int, Array[Float], ArrayIndexItem[Int, Float], Float](dataset)
      case (IntegerType, ArrayType(DoubleType, _)) => typedFit[Int, Array[Double], ArrayIndexItem[Int, Double], Double](dataset)
      case (IntegerType, t) if t.typeName == "vector" => typedFit[Int, Vector, VectorIndexItem[Int], Double](dataset)
      case (LongType, ArrayType(FloatType, _)) => typedFit[Long, Array[Float], ArrayIndexItem[Long, Float], Float](dataset)
      case (LongType, ArrayType(DoubleType, _)) => typedFit[Long, Array[Double], ArrayIndexItem[Long, Double], Double](dataset)
      case (LongType, t) if t.typeName == "vector" => typedFit[Long, Vector, VectorIndexItem[Long], Double](dataset)
      case (StringType, ArrayType(FloatType, _)) => typedFit[String, Array[Float], ArrayIndexItem[String, Float], Float](dataset)
      case (StringType, ArrayType(DoubleType, _)) => typedFit[String, Array[Double], ArrayIndexItem[String, Double], Double](dataset)
      case (StringType, t) if t.typeName == "vector" => typedFit[String, Vector, VectorIndexItem[String], Double](dataset)
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
  ](dimensions: Int, maxItemCount: Int, distanceFunction: DistanceFunction[TVector, TDistance])(implicit distanceOrdering: Ordering[TDistance])
      : TIndex[TId, TVector, TItem, TDistance]

  /**
    * Creates the model to be returned from fitting the data.
    *
    * @param uid identifier
    * @param indices rdd that holds the partitioned indices that are used to do the search
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
  ](uid: String, indices: RDD[(Int, TIndex[TId, TVector, TItem, TDistance])])(implicit ev: ClassTag[TId], evVector: ClassTag[TVector], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance])
    : TModel

  private def typedFit[
    TId : TypeTag,
    TVector : TypeTag,
    TItem <: Item[TId, TVector] with Product : TypeTag,
    TDistance: TypeTag
  ](dataset: Dataset[_])(implicit ev: ClassTag[TId], evVector: ClassTag[TVector], evItem: ClassTag[TItem], evDistance: ClassTag[TDistance], distanceOrdering: Ordering[TDistance], distanceFunctionFactory: String => DistanceFunction[TVector, TDistance])
    : TModel = {

    import dataset.sparkSession.implicits._

    val items = dataset.select(col(getIdentifierCol).as("id"), col(getFeaturesCol).as("vector"))
      .as[TItem]

    val storageLevel = StorageLevel.fromString(getStorageLevel)

    val partitioner = new PartitionIdPassthrough(getNumPartitions)

    // read the id and vector from the input dataset and and repartition them over numPartitions amount of partitions.
    // Transform vectors or double arrays into float arrays for performance reasons.

    val partitionedIndexItems = items
      .mapPartitions { _.map (item => (abs(item.id.hashCode) % getNumPartitions, item)) }
      .rdd
      .partitionBy(partitioner)

    // On each partition collect all the items into memory and construct the HNSW indices.
    // The result is a rdd that has a single row per partition containing the index

    val indicesRdd = partitionedIndexItems
      .mapPartitionsWithIndex((partition, it) =>
        if (it.hasNext) {
          val items = it.map { case (_, indexItem) => indexItem }.toList

          logInfo(f"partition $partition%04d: indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          val index = createIndex[TId, TVector, TItem, TDistance](items.head.dimensions, items.size, distanceFunctionFactory(getDistanceFunction))
          index.addAll(items, progressUpdateInterval = 5000, listener = (workDone, max) => logDebug(f"partition $partition%04d: Indexed $workDone of $max items"))

          logInfo(f"partition $partition%04d: done indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          Iterator.single(partition -> index)
        } else Iterator.empty
        , preservesPartitioning = true).persist(storageLevel)

    if (storageLevel != StorageLevel.NONE) {
      // force caching
      indicesRdd.count()
    }

    createModel[TId, TVector, TItem, TDistance](uid, indicesRdd)
  }

  implicit private def floatArrayDistanceFunction(name: String): DistanceFunction[Array[Float], Float] = name match {
    case "bray-curtis" => floatBrayCurtisDistance
    case "canberra" => floatCanberraDistance
    case "correlation" => floatCorrelationDistance
    case "cosine" => floatCosineDistance
    case "euclidean" => floatEuclideanDistance
    case "inner-product" => floatInnerProduct
    case "manhattan" => floatManhattanDistance
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Array[Float] @unchecked, Float @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }

  implicit private def doubleArrayDistanceFunction(name: String): DistanceFunction[Array[Double], Double] = name match {
    case "bray-curtis" => doubleBrayCurtisDistance
    case "canberra" => doubleCanberraDistance
    case "correlation" => doubleCorrelationDistance
    case "cosine" => doubleCosineDistance
    case "euclidean" => doubleEuclideanDistance
    case "inner-product" => doubleInnerProduct
    case "manhattan" => doubleManhattanDistance
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Array[Double] @unchecked, Double @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance function."))
  }

  implicit private def vectorDistanceFunction(name: String): DistanceFunction[Vector, Double] = name match {
    case "bray-curtis" => VectorDistanceFunctions.brayCurtisDistance
    case "canberra" => VectorDistanceFunctions.canberraDistance
    case "correlation" => VectorDistanceFunctions.correlationDistance
    case "cosine" => VectorDistanceFunctions.cosineDistance
    case "euclidean" => VectorDistanceFunctions.euclideanDistance
    case "inner-product" => VectorDistanceFunctions.innerProduct
    case "manhattan" => VectorDistanceFunctions.manhattanDistance
    case value =>
      Try(Class.forName(value).getDeclaredConstructor().newInstance())
        .toOption
        .collect { case f: DistanceFunction[Vector @unchecked, Double @unchecked] => f }
        .getOrElse(throw new IllegalArgumentException(s"$value is not a valid distance functions."))
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


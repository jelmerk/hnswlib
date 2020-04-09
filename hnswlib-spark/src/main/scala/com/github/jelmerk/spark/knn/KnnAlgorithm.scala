package com.github.jelmerk.spark.knn

import java.lang.{Float => JFloat}
import java.net.InetAddress

import scala.math.abs
import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.github.jelmerk.knn.scalalike._
import com.github.jelmerk.spark.util.{BoundedPriorityQueue, PartitionedRdd, UnsplittableSequenceFileInputFormat, Utils}
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{BytesWritable, NullWritable}
import org.apache.spark.Partitioner
import org.apache.spark.ml.util.{MLReader, MLWriter}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods._
import org.json4s._

private[knn] case class Query[TVector](id: String, vector: TVector)


/**
  * Neighbor of an item
  *
  * @param neighbor identifies the neighbor
  * @param distance distance to the item
  */
private[knn] case class Neighbor[TId] (neighbor: TId, distance: Float) extends Comparable[Neighbor[TId]] {
  override def compareTo(other: Neighbor[TId]): Int = JFloat.compare(other.distance, distance)
}

/**
  * Common params for KnnAlgorithm and KnnModel.
  */
private[knn] trait KnnModelParams extends Params {

  /**
    * Param for the column name for the row identifier.
    * Default: "id"
    *
    * @group param
    */
  val identifierCol = new Param[String](this, "identifierCol", "column names for the row identifier")

  /** @group getParam */
  def getIdentifierCol: String = $(identifierCol)

  /**
    * Param for the column name for the vector.
    * Default: "vector"
    *
    * @group param
    */
  val vectorCol = new Param[String](this, "vectorCol", "column names for the vector")

  /** @group getParam */
  def getVectorCol: String = $(vectorCol)

  /**
    * Param for the column name for returned neighbors.
    * Default: "neighbors"
    *
    * @group param
    */
  val neighborsCol = new Param[String](this, "neighborsCol", "column names for returned neighbors")

  /** @group getParam */
  def getNeighborsCol: String = $(neighborsCol)

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
    * Param that indicates whether to not return the row identifier as a candidate neighbor.
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
  val similarityThreshold = new FloatParam(this, "similarityThreshold", "do not return neighbors further away than this distance")

  /** @group getParam */
  def getSimilarityThreshold: Float = $(similarityThreshold)

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

  setDefault(k -> 5, neighborsCol -> "neighbors", identifierCol -> "id", vectorCol -> "vector",
    excludeSelf -> false, similarityThreshold -> -1, outputFormat -> "full")

  protected def validateAndTransformSchema(schema: StructType): StructType = {

    val identifierColSchema = schema(getIdentifierCol)

    val neighborsField = StructField(getNeighborsCol, ArrayType(StructType(Seq(StructField("neighbor", identifierColSchema.dataType, identifierColSchema.nullable), StructField("distance", FloatType)))))

    getOutputFormat match {
      case "minimal" => StructType(Array(identifierColSchema, neighborsField))

      case _ =>
        if (schema.fieldNames.contains(getNeighborsCol)) {
          throw new IllegalArgumentException(s"Output column $getNeighborsCol already exists.")
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
    * Param for StorageLevel for the indices. Pass in a string representation of `StorageLevel`.
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
  * @tparam TModel type of the model
  * @tparam TId type of the index item identifier
  * @tparam TVector type of the index item vector
  * @tparam TItem type of the index item
  * @tparam TIndex type of the index
  */
private[knn] class KnnModelWriter[TModel <: Model[TModel],
                                  TId: TypeTag,
                                  TVector : TypeTag,
                                  TItem <: Item[TId, TVector] with Product : TypeTag,
                                  TIndex <: Index[TId, TVector, TItem, Float]]
    (instance: KnnModel[TModel, TId, TVector, TItem, TIndex])
  extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val metaData = JObject(
      JField("class", JString(instance.getClass.getName)),
      JField("timestamp", JInt(System.currentTimeMillis())),
      JField("sparkVersion", JString(sc.version)),
      JField("uid", JString(instance.uid)),
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
  * @param ev classtag of the model type
  * @tparam TModel type of the model
  * @tparam TId type of the index item identifier
  * @tparam TVector type of the index item vector
  * @tparam TItem type of the index item
  * @tparam TIndex type of the index
  */
private[knn] abstract class KnnModelReader[TModel <: Model[TModel],
                                           TId: TypeTag,
                                           TVector : TypeTag,
                                           TItem <: Item[TId, TVector] with Product : TypeTag,
                                           TIndex <: Index[TId, TVector, TItem, Float]]
  (implicit ev: ClassTag[TModel]) extends MLReader[TModel] {

  private implicit val format: Formats = DefaultFormats

  override def load(path: String): TModel = {

    val metadataPath = new Path(path, "metadata").toString

    val metadataStr = sc.textFile(metadataPath, 1).first()

    // cannot use parse because of incompatibilities between json4s 3.2.11 used by spark 2.3 and 3.6.6 used by spark 2.4
    val metadata = mapper.readValue(metadataStr, classOf[JValue])

    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]

    val paramMap = (metadata \ "paramMap").extract[JObject]

    val expectedClassName = ev.runtimeClass.getName

    require(className == expectedClassName, s"Error loading metadata: Expected class name" +
      s" $expectedClassName but found class name $className")

    val indicesPath = new Path(path, "indices").toString

    val indices = sc.hadoopFile(indicesPath, classOf[UnsplittableSequenceFileInputFormat[NullWritable, BytesWritable]], classOf[NullWritable], classOf[BytesWritable])
      .flatMap { case (_, value) => Utils.deserialize[Array[(Int, (TIndex, TId, TVector))]](value.getBytes) }

    val model = createModel(uid, new PartitionedRdd(indices, Some(new PartitionIdPassthrough(indices.getNumPartitions))))

    paramMap.obj.foreach { case (paramName, jsonValue) =>
      val param = model.getParam(paramName)
      model.set(param, param.jsonDecode(compact(render(jsonValue))))
    }

    model
  }

  protected def createModel(uid: String, indices: RDD[(Int, (TIndex, TId, TVector))]): TModel
}

/**
  * Base class for nearest neighbor search models.
  *
  * @param uid identifier
  * @param indices rdd that holds the indices that are used to do the search
  * @param evIdentifier classtag of the index item identifier
  * @tparam TModel type of the model
  * @tparam TId type of the index item identifier
  * @tparam TVector type of the index item vector
  * @tparam TItem type of the index item
  * @tparam TIndex type of the index
  */
private[knn] abstract class KnnModel[TModel <: Model[TModel],
                                     TId : TypeTag,
                                     TVector : TypeTag,
                                     TItem <: Item[TId, TVector] with Product : TypeTag,
                                     TIndex <: Index[TId, TVector, TItem, Float]]
    (override val uid: String, private[knn] val indices: RDD[(Int, (TIndex, TId, TVector))])(implicit evIdentifier: ClassTag[TId])
      extends Model[TModel] with KnnModelParams {

  /** @group setParam */
  def setIdentifierCol(value: String): this.type = set(identifierCol, value)

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  /** @group setParam */
  def setSimilarityThreshold(value: Float): this.type = set(similarityThreshold, value)

  /** @group setParam */
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val identifierType = dataset.schema(getIdentifierCol).dataType

    // read the items and duplicate the rows in the query dataset with the number of partitions, assign a different partition to each copy

    val queryRdd = readQueries(dataset)
      .withColumn("partition", explode(array(0 until indices.getNumPartitions map lit: _*)))
      .as[(TId, TVector, Int)]
      .rdd
      .map { case (id, vector, partition) => (partition, (null.asInstanceOf[TIndex], id, vector)) }
      .partitionBy(indices.partitioner.get)

    // combine the indices rdd and query rdds into a single rdd and make sure the first row of the unioned rdd is our index

    val unioned = indices
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
                    if (!getExcludeSelf || item.id != id) && (getSimilarityThreshold < 0 || distance < getSimilarityThreshold)  =>
                      Neighbor(item.id, distance)
                  }

                val queue = new BoundedPriorityQueue[Neighbor[TId]](getK)
                queue ++= neighbors

                id -> queue
              }
            }
          )
        }
      } else Iterator.empty
    }

    // reduce the top k neighbors on each shard to the top k neighbors over all shards, holding on to only the best matches

    val topNeighbors = neighborsOnAllShards
      .reduceByKey { case (neighborsA, neighborsB) =>
        neighborsA ++= neighborsB
        neighborsA
      }
      .mapValues(_.toArray.sorted(Ordering[Neighbor[TId]].reverse))

    // transform the rdd into our output dataframe

    val transformed = topNeighbors
      .toDF(getIdentifierCol, getNeighborsCol)
      .select(
        col(getIdentifierCol).cast(identifierType).as(getIdentifierCol),
        col(getNeighborsCol).cast(ArrayType(StructType(Seq(
          StructField("neighbor", identifierType),
          StructField("distance", FloatType)
        )))).as(getNeighborsCol)
      )

    if (getOutputFormat == "minimal") transformed
    else dataset.join(transformed, Seq(getIdentifierCol))
  }

  /**
    * Subclasses can implement this class in order to transform the index before querying.
    *
    * @param index the index to transform
    */
  private[knn] def transformIndex(index: TIndex): Unit = ()

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
    * Read queries from the passed in dataset.
    *
    * @param dataset the dataset to read the items from
    * @return dataset of item
    */
  private[knn] def readQueries(dataset: Dataset[_]): Dataset[Query[TVector]]

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

private[knn] abstract class KnnAlgorithm[TModel <: Model[TModel],
                                         TId : TypeTag,
                                         TVector : TypeTag,
                                         TItem <: Item[TId, TVector] with Product : TypeTag,
                                         TIndex <: Index[TId, TVector, TItem, Float]]
    (override val uid: String)(implicit ev: ClassTag[TItem])
  extends Estimator[TModel] with KnnAlgorithmParams  {

  def setIdentifierCol(value: String): this.type = set(identifierCol, value)

  /** @group setParam */
  def setVectorCol(value: String): this.type = set(vectorCol, value)

  /** @group setParam */
  def setNeighborsCol(value: String): this.type = set(neighborsCol, value)

  /** @group setParam */
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  def setNumPartitions(value: Int): this.type = set(numPartitions, value)

  /** @group setParam */
  def setDistanceFunction(value: String): this.type = set(distanceFunction, value)

  /** @group setParam */
  def setExcludeSelf(value: Boolean): this.type = set(excludeSelf, value)

  /** @group setParam */
  def setSimilarityThreshold(value: Float): this.type = set(similarityThreshold, value)

  /** @group setParam */
  def setOutputFormat(value: String): this.type = set(outputFormat, value)

  /** @group setParam */
  def setStorageLevel(value: String): this.type = set(storageLevel, value)

  override def fit(dataset: Dataset[_]): TModel = {
    import dataset.sparkSession.implicits._

    val storageLevel = StorageLevel.fromString(getStorageLevel)

    val partitioner = new PartitionIdPassthrough(getNumPartitions)

    // read the id and vector from the input dataset and and repartition them over numPartitions amount of partitions.
    // Transform vectors or double arrays into float arrays for performance reasons.

    val partitionedIndexItems = readIndexItems(dataset)
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

          val index = createIndex(items.head.dimensions, items.size)
          index.addAll(items, progressUpdateInterval = 5000, listener = (workDone, max) => logDebug(f"partition $partition%04d: Indexed $workDone of $max items"))

          logInfo(f"partition $partition%04d: done indexing ${items.size} items on host ${InetAddress.getLocalHost.getHostName}")

          Iterator.single(partition -> Tuple3(index, null.asInstanceOf[TId], null.asInstanceOf[TVector]))
        } else Iterator.empty
        , preservesPartitioning = true).persist(storageLevel)

    if (storageLevel != StorageLevel.NONE) {
      // force caching
      indicesRdd.count()
    }

    val model = createModel(uid, indicesRdd)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def copy(extra: ParamMap): Estimator[TModel] = defaultCopy(extra)

  /**
   * Read index items from the passed in dataset.
   *
   * @param dataset the dataset to read the items from
   * @return dataset of item
   */
  private[knn] def readIndexItems(dataset: Dataset[_]): Dataset[TItem]

  /**
    * Create the index used to do the nearest neighbor search.
    *
    * @param dimensions dimensionality of the items stored in the index
    * @param maxItemCount maximum number of items the index can hold
    * @return create an index
    */
  private[knn] def createIndex(dimensions: Int, maxItemCount: Int): TIndex

  /**
    * Creates the model to be returned from fitting the data.
    *
    * @param uid identifier
    * @param indices rdd that holds the indices that are used to do the search
    * @return model
    */
  private[knn] def createModel(uid: String, indices: RDD[(Int, (TIndex, TId, TVector))]): TModel

}

/**
  * Partitioner that uses precomputed partitions
  *
  * @param numPartitions number of partitions
  */
private[knn] class PartitionIdPassthrough(override val numPartitions: Int) extends Partitioner {
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
}
import org.github.jelmerk.knn._
import org.github.jelmerk.knn.bruteforce.BruteForceIndex
import org.github.jelmerk.knn.hnsw.HnswIndex


case class MyWord(id: String, @specialized vector: Array[Float]) extends Item[String, Array[Float]]

object Main {



  def main(args: Array[String]): Unit = {

//    val hnswIndex = HnswIndex.Builder.usingCosineDistance(100000)
//      .withEf(10)
//      .withEfConstruction(200)
//      .withM(16)
//      .build[String, MyWord]()
//
//    val index = BruteForceIndex.Builder.usingCosineDistance()
//      .build[String, MyWord]()

//    val r = index.getOptionally("king")
//      .toSeq
//      .flatMap(word => index.findNearestAsSeq(word.vector, 10))


//    val r = index.findNearestAsSeq(index.get("king").vector, 10)


//    val r = index.findNearest(index.get("king").vector, 10)

//    r.foreach(println)

  }
}

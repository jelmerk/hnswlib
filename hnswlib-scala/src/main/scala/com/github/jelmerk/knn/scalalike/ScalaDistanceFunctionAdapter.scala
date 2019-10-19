package com.github.jelmerk.knn.scalalike

import com.github.jelmerk.knn.{DistanceFunction => JDistanceFunction}

/**
  * Adapts a scala function to the java DistanceFunction interface
  *
  * @param scalaFunction scala function to delegate to
  *
  * @tparam TVector Type of the vector to perform distance calculation on
  * @tparam TDistance Type of distance between items (expect any numeric type: float, double, int, ..)
  */
@SerialVersionUID(1L)
private[scalalike] class ScalaDistanceFunctionAdapter[TVector, TDistance](val scalaFunction: DistanceFunction[TVector, TDistance])
  extends JDistanceFunction[TVector, TDistance] {

  override def distance(u: TVector, v: TVector): TDistance = scalaFunction(u, v)

}

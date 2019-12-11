package com.github.jelmerk.knn.scalalike

object SparseVector {

  /**
    * Extract fields from SparseVector.
    *
    * @param vector vector to extract the fields from
    * @tparam TVector type of values array
    * @return the extracted fields
    */
  def unapply[TVector](vector: SparseVector[TVector]): Option[(Array[Int], TVector)] =
    Some(vector.indices -> vector.values)

  /**
    * SparseVector factory.
    *
    * @param indices the index array, must be in ascending order
    * @param values the values array
    * @tparam TVector type of values array
    * @return a new SparseVector
    */
  def apply[TVector](indices: Array[Int], values: TVector): SparseVector[TVector] =
    new SparseVector[TVector](indices, values)

}

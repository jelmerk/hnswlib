package com.github.jelmerk.knn;

import java.io.Serializable;

/**
 * Calculates distance between 2 vectors.
 *
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 */
@FunctionalInterface
public interface DistanceFunction<TVector, TDistance> extends Serializable {

    /**
     * Gets the distance between 2 items.
     *
     * @param u from item
     * @param v to item
     * @return The distance between items.
     */
    TDistance distance(TVector u, TVector v);

}

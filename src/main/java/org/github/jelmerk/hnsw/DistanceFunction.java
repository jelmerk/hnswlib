package org.github.jelmerk.hnsw;

import java.io.Serializable;

/**
 * Calculates distance between 2 items.
 *
 * @param <TVector>
 * @param <TDistance>
 */
@FunctionalInterface
public interface DistanceFunction<TVector, TDistance extends Comparable<TDistance>> extends Serializable {

    /**
     * Gets the distance between 2 items.
     *
     * @param u from item
     * @param v to item
     * @return The distance between items.
     */
    TDistance distance(TVector u, TVector v);

}

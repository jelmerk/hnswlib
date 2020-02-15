package com.github.jelmerk.knn;

import java.io.Serializable;

/**
 * Indexable item.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 */
public interface Item<TId, TVector> extends Serializable {

    /**
     * Returns the identifier of this item.
     *
     * @return the identifier of this item
     */
    TId id();

    /**
     * Returns the vector to perform the distance calculation on.
     *
     * @return the vector to perform the distance calculation on
     */
    TVector vector();

    /**
     * Returns the dimensionality of the vector.
     *
     * @return the dimensionality of the vector
     */
    int dimensions();

    /**
     * Returns the version of the item. Higher is newer.
     *
     * @return the version of this item.
     */
    default long version() {
        return 0;
    }
}

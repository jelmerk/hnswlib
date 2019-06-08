package com.github.jelmerk.knn;

import java.io.Serializable;

/**
 * Item that can be indexed.
 *
 * @param <TId> The type of the vector to perform distance calculation on
 * @param <TVector> The type of the vector to perform distance calculation on
 */
public interface Item<TId, TVector> extends Serializable {

    /**
     * Returns the identifier of this item.
     *
     * @return the idenifier of this item
     */
    TId id();

    /**
     * Returns the vector to perform the distance calculation on.
     *
     * @return the vector to perform the distance calculation on
     */
    TVector vector();
}

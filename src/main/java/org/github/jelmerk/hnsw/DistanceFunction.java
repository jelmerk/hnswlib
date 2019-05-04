package org.github.jelmerk.hnsw;

import java.io.Serializable;

/**
 * Calculates distance between 2 items.
 *
 * @param <Item>
 */
@FunctionalInterface
public interface DistanceFunction<Item, TDistance extends Comparable<TDistance>> extends Serializable {

    /**
     * Gets the distance between 2 items.
     *
     * @param u from item
     * @param v to item
     * @return The distance between items.
     */
    TDistance distance(Item u, Item v);

}

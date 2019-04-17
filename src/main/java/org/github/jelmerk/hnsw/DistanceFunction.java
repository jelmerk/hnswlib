package org.github.jelmerk.hnsw;

import java.io.Serializable;

/**
 * Calculates distance between 2 items.
 *
 * @param <Item>
 */
public interface DistanceFunction<Item> extends Serializable {

    /**
     * Gets the distance between 2 items.
     *
     * @param u from item
     * @param v to item
     * @return The distance between items.
     */
    float distance(Item u, Item v);

}

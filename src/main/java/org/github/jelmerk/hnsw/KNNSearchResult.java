package org.github.jelmerk.hnsw;

import java.io.Serializable;

/**
 * Representation of knn search result.
 */
public class KNNSearchResult<TItem, TDistance extends Comparable<TDistance>> implements Serializable {

    private int id;
    private TItem item;
    private TDistance distance;

    /**
     * Gets the id of the item = rank of the item in source collection.
     */
    public int getId() {
        return id;
    }

    /**
     * Sets the id of the item = rank of the item in source collection
     */
    public void setId(int id) {
        this.id = id;
    }

    /**
     * Gets the item itself.
     */
    public TItem getItem() {
        return item;
    }

    /**
     * Sets the item itself.
     */
    public void setItem(TItem item) {
        this.item = item;
    }

    /**
     * Gets the distance between the item and the knn search query.
     */
    public TDistance getDistance() {
        return distance;
    }

    /**
     * Sets the distance between the item and the knn search query.
     */
    public void setDistance(TDistance distance) {
        this.distance = distance;
    }

    @Override
    public String toString() {
        return "KNNSearchResult{" +
                "id=" + id +
                ", item=" + item +
                ", distance=" + distance +
                '}';
    }
}
package org.github.jelmerk.knn;

import java.io.Serializable;

public class SearchResult<TItem, TDistance extends Comparable<TDistance>>
        implements Comparable<SearchResult<TItem, TDistance>>, Serializable {

    private static final long serialVersionUID = 1L;

    private final TDistance distance;

    private final TItem item;

    public SearchResult( TItem item, TDistance distance) {
        this.item = item;
        this.distance = distance;
    }

    public TItem getItem() {
        return item;
    }

    public TDistance getDistance() {
        return distance;
    }

    @Override
    public int compareTo(SearchResult<TItem, TDistance> o) {
        return this.distance.compareTo(o.distance);
    }

    @Override
    public String toString() {
        return "SearchResult{" +
                "distance=" + distance +
                ", item=" + item +
                '}';
    }
}
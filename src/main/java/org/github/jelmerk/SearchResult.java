package org.github.jelmerk;

public class SearchResult<TItem, TDistance extends Comparable<TDistance>> {

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
}
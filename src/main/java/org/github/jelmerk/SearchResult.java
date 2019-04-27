package org.github.jelmerk;

public class SearchResult<TItem, TDistance extends Comparable<TDistance>> {

    private final TDistance distance;

    private final TItem item;

    public SearchResult(TDistance distance, TItem item) {
        this.distance = distance;
        this.item = item;
    }

    public TDistance getDistance() {
        return distance;
    }

    public TItem getItem() {
        return item;
    }
}
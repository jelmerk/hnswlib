package org.github.jelmerk.knn;

import java.io.Serializable;
import java.util.Objects;

/**
 * Result of a nearest neighbour search.
 *
 * @param <TItem> type of the item returned
 * @param <TDistance> type of the distance returned by the configured distance function
 */
public class SearchResult<TItem, TDistance extends Comparable<TDistance>>
        implements Comparable<SearchResult<TItem, TDistance>>, Serializable {

    private static final long serialVersionUID = 1L;

    private final TDistance distance;

    private final TItem item;

    /**
     * Constructs a new SearchResult instance.
     *
     * @param item the item
     * @param distance the distance from the search query
     */
    public SearchResult( TItem item, TDistance distance) {
        this.item = item;
        this.distance = distance;
    }

    /**
     * Returns the item.
     *
     * @return the item
     */
    public TItem item() {
        return item;
    }

    /**
     * Returns the distance from the search query.
     *
     * @return the distance from the search query
     */
    public TDistance distance() {
        return distance;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int compareTo(SearchResult<TItem, TDistance> o) {
        return this.distance.compareTo(o.distance);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SearchResult<?, ?> that = (SearchResult<?, ?>) o;
        return Objects.equals(distance, that.distance) &&
                Objects.equals(item, that.item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return Objects.hash(distance, item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return "SearchResult{" +
                "distance=" + distance +
                ", item=" + item +
                '}';
    }
}
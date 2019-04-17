package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.Comparator;

/**
 * Implementation of distance calculation from an arbitrary point to the given destination.
 *
 * @param <TItem> >Type of the points.
 * @param <TDistance> Type of the diatnce.
 */
public class TravelingCosts<TItem, TDistance extends Comparable<TDistance>> implements Comparator<TItem>, Serializable {

    // Default distance comparator.
    private final Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

    // The distance function.
    private final DistanceFunction<TItem, TDistance> distance;

    // he destination point.
    private final TItem destination;

    /**
     * Initializes a new instance of the {@link TravelingCosts} class.
     *
     * @param distance The distance function.
     * @param destination The destination point.
     */
    public TravelingCosts(DistanceFunction<TItem, TDistance> distance, TItem destination) {
        this.distance = distance;
        this.destination = destination;
    }

    /**
     * Gets the distance function.
     */
    public DistanceFunction<TItem, TDistance> getDistance() {
        return distance;
    }

    /**
     * Gets the destination.
     */
    public TItem getDestination() {
        return destination;
    }

    /**
     * Calculates distance from the departure to the destination.
     * @param departure The point of departure.
     *
     * @return The distance from the departure to the destination.
     */
    public TDistance from(TItem departure) {
        return this.distance.distance(departure, this.destination);
    }

    /**
     * Compares 2 points by the distance from the destination.
     *
     * @param x Left point.
     * @param y Right point.
     * @return -1 if x is closer to the destination than y;
     * 0 if x and y are equally far from the destination;
     * 1 if x is farther from the destination than y.
     */
    public int compare(TItem x, TItem y) {
        TDistance fromX = this.from(x);
        TDistance fromY = this.from(y);
        return distanceComparator.compare(fromX, fromY);
    }
}

package org.github.jelmerk.hnsw;

import java.util.Comparator;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

public class TravelingCosts<TItem, TDistance extends Comparable<TDistance>> implements Comparator<TItem> {

    private final BiFunction<TItem, TItem, TDistance> distance;

    private final TItem destination;

    private final Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

    private final ConcurrentHashMap<TItem, TDistance> cache;

    public TravelingCosts(BiFunction<TItem, TItem, TDistance> distance, TItem destination) {
        this.distance = distance;
        this.destination = destination;

        this.cache = new ConcurrentHashMap<>();
    }

    public BiFunction<TItem, TItem, TDistance> getDistance() {
        return distance;
    }

    public TItem getDestination() {
        return destination;
    }

    public TDistance from(TItem departure) {
        TDistance result = cache.get(departure);
        if (result == null) {
            result = distance.apply(departure, destination);
            this.cache.put(departure, result);
        }
        return result;
    }

    public int compare(TItem x, TItem y) {
        TDistance fromX = this.from(x);
        TDistance fromY = this.from(y);
        return distanceComparator.compare(fromX, fromY);
    }
}

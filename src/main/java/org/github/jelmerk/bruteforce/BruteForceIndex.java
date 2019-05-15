package org.github.jelmerk.bruteforce;

import org.github.jelmerk.Index;
import org.github.jelmerk.Item;
import org.github.jelmerk.SearchResult;
import org.github.jelmerk.hnsw.DistanceFunction;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class BruteForceIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private final DistanceFunction<TVector, TDistance> distanceFunction;
    private final Map<TId, TItem> items;

    public BruteForceIndex(BruteForceIndex.Builder<TVector, TDistance> builder) {
        this.distanceFunction = builder.distanceFunction;
        this.items = new ConcurrentHashMap<>();
    }

    @Override
    public int size() {
        return items.size();
    }

    @Override
    public TItem get(TId id) {
        return items.get(id);
    }

    @Override
    public void add(TItem item) {
        items.putIfAbsent(item.getId(), item);
    }

    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector tVector, int k) {

        Comparator<SearchResult<TItem, TDistance>> comparator = Comparator
                .<SearchResult<TItem, TDistance>>naturalOrder()
                .reversed();

        PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);

        for (TItem item : items.values()) {
            TDistance distance = distanceFunction.distance(item.getVector(), tVector);

            SearchResult<TItem, TDistance> searchResult = new SearchResult<>(item, distance);
            queue.add(searchResult);

            if (queue.size() > k) {
                queue.poll();
            }
        }

        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(queue.size());

        SearchResult<TItem, TDistance> result;
        while((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
            results.add(0, result);
        }

        return results;
    }

    public static class Builder <TVector, TDistance extends Comparable<TDistance>> {

        private final DistanceFunction<TVector, TDistance> distanceFunction;

        public Builder(DistanceFunction<TVector, TDistance> distanceFunction) {
            this.distanceFunction = distanceFunction;
        }

        public <TId, TItem extends Item<TId, TVector>> BruteForceIndex<TId, TVector, TItem, TDistance> build() {
            return new BruteForceIndex<>(this);
        }
    }

}

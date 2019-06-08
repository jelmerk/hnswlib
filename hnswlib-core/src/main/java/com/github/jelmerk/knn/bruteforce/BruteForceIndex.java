package com.github.jelmerk.knn.bruteforce;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of {@link Index} that does pairwise comparison and as such can be used as a baseline for measuring
 * approximate nearest neighbours index accuracy.
 *
 * @param <TId> type of the external identifier of an item
 * @param <TVector> The type of the vector to perform distance calculation on
 * @param <TItem> The type of items to connect into small world.
 * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
 */
public class BruteForceIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private static final long serialVersionUID = 1L;

    private final DistanceFunction<TVector, TDistance> distanceFunction;
    private final Comparator<TDistance> distanceComparator;

    private final Map<TId, TItem> items;

    private BruteForceIndex(BruteForceIndex.Builder<TVector, TDistance> builder) {
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.items = new ConcurrentHashMap<>();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return items.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        return Optional.ofNullable(items.get(id));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(TItem item) {
        items.putIfAbsent(item.id(), item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId tId) {
        return items.remove(tId) != null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {

        Comparator<SearchResult<TItem, TDistance>> comparator = Comparator
                .<SearchResult<TItem, TDistance>>naturalOrder()
                .reversed();

        PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);

        for (TItem item : items.values()) {
            TDistance distance = distanceFunction.distance(item.vector(), vector);

            SearchResult<TItem, TDistance> searchResult = new SearchResult<>(item, distance, distanceComparator);
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

    /**
     * Restores a {@link BruteForceIndex} instance from a file created by invoking the
     * {@link BruteForceIndex#save(File)} method.
     *
     * @param file file to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        BruteForceIndex<TId, TVector, TItem, TDistance>
            load(File file) throws IOException {

        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link BruteForceIndex} instance from a file created by invoking the {@link BruteForceIndex#save(Path)} method.
     *
     * @param path path to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        BruteForceIndex<TId, TVector, TItem, TDistance>
            load(Path path) throws IOException {

        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link BruteForceIndex} instance from a file created by invoking the
     * {@link BruteForceIndex#save(File)} method.
     *
     * @param inputStream InputStream to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        BruteForceIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream) throws IOException {

        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (BruteForceIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    public static <TVector, TDistance extends Comparable<TDistance>>
        Builder <TVector, TDistance> newBuilder(DistanceFunction<TVector, TDistance> distanceFunction) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();
        return new Builder<>(distanceFunction, distanceComparator);
    }

    public static <TVector, TDistance>
        Builder <TVector, TDistance>
            newBuilder(DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator) {

        return new Builder<>(distanceFunction, distanceComparator);
    }

    /**
     * Builder for initializing an {@link BruteForceIndex} instance.
     *
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     */
    public static class Builder <TVector, TDistance> {

        private final DistanceFunction<TVector, TDistance> distanceFunction;

        private final Comparator<TDistance> distanceComparator;

        Builder(DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator) {
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
        }

        /**
         * Builds the BruteForceIndex instance.
         *
         * @param <TId> type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the brute force index instance
         */
        public <TId, TItem extends Item<TId, TVector>> BruteForceIndex<TId, TVector, TItem, TDistance> build() {
            return new BruteForceIndex<>(this);
        }

    }

}

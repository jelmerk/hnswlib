package com.github.jelmerk.knn.bruteforce;

import com.github.jelmerk.knn.DistanceFunction;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.Item;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.util.ClassLoaderObjectInputStream;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Implementation of {@link Index} that does pairwise comparison and as such can be used as a baseline for measuring
 * approximate nearest neighbors index precision.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 */
public class BruteForceIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance> {

    private static final long serialVersionUID = 1L;

    private final int dimensions;
    private final DistanceFunction<TVector, TDistance> distanceFunction;
    private final Comparator<TDistance> distanceComparator;

    private final Map<TId, TItem> items;
    private final Map<TId, Long> deletedItemVersions;

    private BruteForceIndex(BruteForceIndex.Builder<TVector, TDistance> builder) {
        this.dimensions = builder.dimensions;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;
        this.items = new ConcurrentHashMap<>();
        this.deletedItemVersions = new ConcurrentHashMap<>();
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
    public Collection<TItem> items() {
        return items.values();
    }

    /**
     * Returns the dimensionality of the items stored in this index.
     *
     * @return the dimensionality of the items stored in this index
     */
    public int getDimensions() {
        return dimensions;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean add(TItem item) {
        if (item.dimensions() != dimensions) {
            throw new IllegalArgumentException("Item does not have dimensionality of : " + dimensions);
        }
        synchronized (items) {
            TItem existingItem = items.get(item.id());

            if (existingItem != null && item.version() < existingItem.version()) {
                return false;
            }

            if (item.version() < deletedItemVersions.getOrDefault(item.id(), 0L)) {
                return false;
            }

            items.put(item.id(), item);
            return true;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id, long version) {
        synchronized (items) {
            TItem item = items.get(id);

            if (item == null) {
                return false;
            }

            if (version < item.version()) {
                return false;
            }
            items.remove(id);
            deletedItemVersions.put(id, version);

            return true;
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {
        if (vector == null) {
            throw new IllegalArgumentException("Vector cannot be null.");
        }

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
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Restores a {@link BruteForceIndex} from a File.
     *
     * @param file file to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link BruteForceIndex} from a File.
     *
     * @param file file to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(File file, ClassLoader classLoader) throws IOException {
        return load(new FileInputStream(file), classLoader);
    }

    /**
     * Restores a {@link BruteForceIndex} from a Path.
     *
     * @param path path to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(Path path) throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link BruteForceIndex} from a Path.
     *
     * @param path path to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(Path path, ClassLoader classLoader) throws IOException {
        return load(Files.newInputStream(path), classLoader);
    }

    /**
     * Restores a {@link BruteForceIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream) throws IOException {
        return load(inputStream, Thread.currentThread().getContextClassLoader());
    }

    /**
     * Restores a {@link BruteForceIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param classLoader the classloader to use
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> BruteForceIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream, ClassLoader classLoader) throws IOException {

        try(ObjectInputStream ois = new ClassLoaderObjectInputStream(classLoader, inputStream)) {
            return (BruteForceIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    /**
     * Start the process of building a new BruteForce index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance extends Comparable<TDistance>>
        Builder <TVector, TDistance> newBuilder(int dimensions, DistanceFunction<TVector, TDistance> distanceFunction) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();
        return new Builder<>(dimensions, distanceFunction, distanceComparator);
    }

    /**
     * Start the process of building a new BruteForce index.
     *
     * @param dimensions the dimensionality of the vectors stored in the index
     * @param distanceFunction the distance function
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance> Builder <TVector, TDistance> newBuilder(int dimensions, DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator) {

        return new Builder<>(dimensions, distanceFunction, distanceComparator);
    }

    /**
     * Builder for initializing an {@link BruteForceIndex} instance.
     *
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder <TVector, TDistance> {

        private final int dimensions;

        private final DistanceFunction<TVector, TDistance> distanceFunction;

        private final Comparator<TDistance> distanceComparator;

        Builder(int dimensions, DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator) {
            this.dimensions = dimensions;
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
        }

        /**
         * Builds the BruteForceIndex instance.
         *
         * @param <TId> Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the brute force index instance
         */
        public <TId, TItem extends Item<TId, TVector>> BruteForceIndex<TId, TVector, TItem, TDistance> build() {
            return new BruteForceIndex<>(this);
        }

    }

}

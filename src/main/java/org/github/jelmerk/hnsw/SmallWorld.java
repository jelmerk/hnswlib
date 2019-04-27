package org.github.jelmerk.hnsw;

import java.io.*;
import java.util.*;

/**
 * The Hierarchical Navigable Small World Graphs.
 *
 * @see <a href="https://arxiv.org/abs/1603.09320">Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 * @param <TItem> The type of items to connect into small world.
 * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
 */
public class SmallWorld<TItem, TDistance extends Comparable<TDistance>> implements Serializable {

    // The distance function in the items space.
    private DistanceFunction<TItem, TDistance> distance;

    // The hierarchical small world graph instance.
    private Graph<TItem, TDistance> graph;

    /**
     * Initializes a new instance of the {@link SmallWorld} class.
     *
     * @param distance The distance function to use in the small world.
     */
    public SmallWorld(DistanceFunction<TItem, TDistance> distance) {
        this.distance = distance;
    }

    /**
     * Builds hnsw graph from the items.
     *
     * @param items The items to connect into the graph.
     * @param generator The random number generator for building graph.
     * @param parameters Parameters of the algorithm.
     */
    public void buildGraph(List<TItem> items, DotNetRandom generator, Parameters parameters) {
        Graph<TItem, TDistance> graph = new Graph<>(this.distance, parameters);
        graph.build(items, generator);
        this.graph = graph;
    }

    /**
     * Run knn search for a given item.
     *
     * @param item The item to search nearest neighbours.
     * @param k The number of nearest neighbours.
     * @return The list of found nearest neighbours.
     */
    public List<KNNSearchResult<TItem, TDistance>> knnSearch(TItem item, int k) {
        return this.graph.kNearest(item, k);
    }

    /**
     * Prints edges of the graph.
     * Mostly for debug and test purposes.
     *
     * @return String representation of the graph's edges.
     */
    public String print() {
        return this.graph.print();
    }

    /**
     * Saves the small world to disk.
     *
     * @param file file to write to
     * @throws IOException in case of an I/O exception
     */
    public void save(File file) throws IOException {
        save(new FileOutputStream(file));
    }

    /**
     * Saves the small world to the passed in OutputStream.
     *
     * @param out OutputStream to write to
     * @throws IOException in case of an I/O exception
     */
    public void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Restores a {@link SmallWorld} instance from a file created by invoking the {@link SmallWorld#save(File)} method.
     *
     * @param file file to initialize the small world from
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <TItem, TDistance extends Comparable<TDistance>> SmallWorld<TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link SmallWorld} instance from a file created by invoking the {@link SmallWorld#save(File)} method.
     *
     * @param inputStream InputStream to initialize the small world from
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TItem, TDistance extends Comparable<TDistance>> SmallWorld<TItem, TDistance> load(InputStream inputStream) throws IOException {
        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (SmallWorld<TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

}

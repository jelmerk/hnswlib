package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.*;

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
     * Type of heuristic to select best neighbours for a node.
     */
    enum NeighbourSelectionHeuristic {
        /**
         * Marker for the Algorithm 3 (SELECT-NEIGHBORS-SIMPLE) from the article.
         * Implemented in {@link org.github.jelmerk.hnsw.Node.Algorithm3}
         */
        SELECT_SIMPLE,

        /**
         * Marker for the Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) from the article.
         * Implemented in {@link org.github.jelmerk.hnsw.Node.Algorithm4}
         */
        SELECT_HEURISTIC
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
     * Parameters of the algorithm.
     */
    public static class Parameters implements Serializable {

        private int m;
        private double levelLambda;
        private NeighbourSelectionHeuristic neighbourHeuristic;
        private int constructionPruning;
        private boolean expandBestSelection;
        private boolean keepPrunedConnections;
        private boolean enableDistanceCacheForConstruction;

        /**
         * Initializes a new instance of the {@link Parameters} class.
         */
        public Parameters() {
            this.m = 10;
            this.levelLambda = 1 / Math.log(this.m);
            this.neighbourHeuristic = NeighbourSelectionHeuristic.SELECT_SIMPLE;
            this.constructionPruning = 200;
            this.expandBestSelection = false;
            this.keepPrunedConnections = true;
            this.enableDistanceCacheForConstruction = true;
        }

        /**
         * Gets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
         *
         * The maximum number of neighbors for the zero layer is 2 * M.
         * The maximum number of neighbors for higher layers is M.
         */
        public int getM() {
            return m;
        }

        /**
         * Sets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
         *
         * The maximum number of neighbors for the zero layer is 2 * M.
         * The maximum number of neighbors for higher layers is M.
         */
        public void setM(int m) {
            this.m = m;
        }

        /**
         * Gets the max level decay parameter.
         *
         * @see <a href="https://en.wikipedia.org/wiki/Exponential_distribution">exponential distribution on wikipedia</a>
         * @see "'mL' parameter in the HNSW article."
         */
        public double getLevelLambda() {
            return levelLambda;
        }

        /**
         * Sets the max level decay parameter.
         *
         * @see <a href="https://en.wikipedia.org/wiki/Exponential_distribution">exponential distribution on wikipedia</a>
         * @see "'mL' parameter in the HNSW article."
         */
        public void setLevelLambda(double levelLambda) {
            this.levelLambda = levelLambda;
        }

        /**
         * Gets parameter which specifies the type of heuristic to use for best neighbours selection.
         */
        public NeighbourSelectionHeuristic getNeighbourHeuristic() {
            return neighbourHeuristic;
        }

        /**
         * Sets parameter which specifies the type of heuristic to use for best neighbours selection.
         */
        public void setNeighbourHeuristic(NeighbourSelectionHeuristic neighbourHeuristic) {
            this.neighbourHeuristic = neighbourHeuristic;
        }

        /**
         * Gets the number of candidates to consider as neighbours for a given node at the graph construction phase.
         *
         * @see "'efConstruction' parameter in the article."
         */
        public int getConstructionPruning() {
            return constructionPruning;
        }

        /**
         * Sets the number of candidates to consider as neighbours for a given node at the graph construction phase.
         *
         * @see "'efConstruction' parameter in the article."
         */
        public void setConstructionPruning(int constructionPruning) {
            this.constructionPruning = constructionPruning;
        }

        /**
         * Gets a value indicating whether to expand candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
         *
         * @see "'extendCandidates' parameter in the article."
         */
        public boolean isExpandBestSelection() {
            return expandBestSelection;
        }

        /**
         * Sets a value indicating whether to expand candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
         *
         * @see "'extendCandidates' parameter in the article."
         */
        public void setExpandBestSelection(boolean expandBestSelection) {
            this.expandBestSelection = expandBestSelection;
        }

        /**
         * Gets a value indicating whether to keep pruned candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
         *
         * @see "'keepPrunedConnections' parameter in the article."
         */
        public boolean isKeepPrunedConnections() {
            return keepPrunedConnections;
        }

        /**
         * Sets a value indicating whether to keep pruned candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
         *
         * @see "'keepPrunedConnections' parameter in the article."
         */
        public void setKeepPrunedConnections(boolean keepPrunedConnections) {
            this.keepPrunedConnections = keepPrunedConnections;
        }

        /**
         * Gets a value indicating whether to cache calculated distances at graph construction time.
         */
        public boolean isEnableDistanceCacheForConstruction() {
            return enableDistanceCacheForConstruction;
        }

        /**
         * Sets a value indicating whether to cache calculated distances at graph construction time.
         */
        public void setEnableDistanceCacheForConstruction(boolean enableDistanceCacheForConstruction) {
            this.enableDistanceCacheForConstruction = enableDistanceCacheForConstruction;
        }
    }

    /**
     * Representation of knn search result.
     */
    static class KNNSearchResult<TItem, TDistance extends Comparable<TDistance>> implements Serializable {

        private int id;
        private TItem item;
        private TDistance distance;

        /**
         * Gets the id of the item = rank of the item in source collection.
         */
        public int getId() {
            return id;
        }

        /**
         * Sets the id of the item = rank of the item in source collection
         */
        public void setId(int id) {
            this.id = id;
        }

        /**
         * Gets the item itself.
         */
        public TItem getItem() {
            return item;
        }

        /**
         * Sets the item itself.
         */
        public void setItem(TItem item) {
            this.item = item;
        }

        /**
         * Gets the distance between the item and the knn search query.
         */
        public TDistance getDistance() {
            return distance;
        }

        /**
         * Sets the distance between the item and the knn search query.
         */
        public void setDistance(TDistance distance) {
            this.distance = distance;
        }

        @Override
        public String toString() {
            return "KNNSearchResult{" +
                    "id=" + id +
                    ", item=" + item +
                    ", distance=" + distance +
                    '}';
        }
    }

}

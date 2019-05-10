package org.github.jelmerk.hnsw;


import org.eclipse.collections.api.list.MutableList;
import org.eclipse.collections.api.list.primitive.IntList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.set.primitive.MutableIntSet;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;
import org.github.jelmerk.Index;
import org.github.jelmerk.Item;
import org.github.jelmerk.SearchResult;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;

public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private final DotNetRandom random;

    private final DistanceFunction<TVector, TDistance> distanceFunction;

    private final int maxItemCount;
    private final int m;
    private final double levelLambda;
    private final int constructionPruning;
    private final boolean expandBestSelection;
    private final boolean keepPrunedConnections;

    private final AtomicInteger itemCount;
    private AtomicReferenceArray<TItem> items;
    private AtomicReferenceArray<Node> nodes;

    private final Map<TId, TItem> lookup;

    private Algorithm algorithm;

    private volatile Node entryPoint;

    private ReentrantLock globalLock;

    private Pool<VisitedBitSet> visitedBitSetPool;
    private Pool<MutableIntList> expansionBufferPool;


    private HnswIndex(HnswIndex.Builder<TVector, TDistance> builder) {

        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.m = builder.m;
        this.levelLambda = builder.levelLambda;
        this.constructionPruning = builder.constructionPruning;
        this.expandBestSelection = builder.expandBestSelection;
        this.keepPrunedConnections =  builder.keepPrunedConnections;

        if (builder.neighbourHeuristic == NeighbourSelectionHeuristic.SELECT_SIMPLE) {
            this.algorithm = new Algorithm3();
        } else {
            this.algorithm = new Algorithm4();
        }

        this.random = new DotNetRandom(builder.randomSeed);

        this.globalLock = new ReentrantLock();

        this.itemCount = new AtomicInteger();
        this.items = new AtomicReferenceArray<>(this.maxItemCount);
        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ConcurrentHashMap<>();

        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount), 12);
        this.expansionBufferPool = new Pool<>(IntArrayList::new, 12);
    }

    @Override
    public int size() {
        return itemCount.get();
    }

    @Override
    public Collection<TItem> items() {
        return lookup.values();
    }

    @Override
    public TItem get(TId id) {
        return lookup.get(id);
    }

    @Override
    public void add(TItem item) {

        int count = itemCount.getAndIncrement(); // TODO JK i guess we don't want this count to increase if there's no space, how ?

        if (count >= this.maxItemCount) {
            throw new IllegalStateException("The number of elements exceeds the specified limit.");
        }

        items.set(count, item);

        Node newNode = this.algorithm.newNode(count, randomLayer(random, this.levelLambda));
        nodes.set(count, newNode);

        lookup.put(item.getId(), item);

        globalLock.lock();

        if (this.entryPoint == null) {
            this.entryPoint = newNode;
        }

        Node entrypointCopy = entryPoint;

        if (newNode.maxLayer() <= entryPoint.maxLayer()) {
            globalLock.unlock();
        }

        try {
            synchronized (newNode) {

                // zoom in and find the best peer on the same level as newNode
                TravelingCosts<Integer, TDistance> currentNodeTravelingCosts = new TravelingCosts<>(this::calculateDistance, newNode.id);

                int bestPeerId = findBestPeer(entrypointCopy.id, entrypointCopy.maxLayer(), newNode.maxLayer(), currentNodeTravelingCosts);


                MutableIntList neighboursIdsBuffer = new IntArrayList(algorithm.getM(0) + 1);

                // connecting new node to the small world
                for (int layer = Math.min(newNode.maxLayer(), entrypointCopy.maxLayer()); layer >= 0; layer--) {
                    runKnnAtLayer(bestPeerId, currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.constructionPruning);
                    MutableIntList bestNeighboursIds = algorithm.selectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

                    for (int i = 0; i < bestNeighboursIds.size(); i++) {
                        int newNeighbourId = bestNeighboursIds.get(i);

                        Node neighbourNode;
                        synchronized (neighbourNode = nodes.get(newNeighbourId)) {
                            algorithm.connect(newNode, neighbourNode, layer); // JK this can mutate the new node
                            algorithm.connect(neighbourNode, newNode, layer); // JK this can mutat the neighbour node
                        }

                        // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                        if (DistanceUtils.lt(currentNodeTravelingCosts.from(newNeighbourId), currentNodeTravelingCosts.from(bestPeerId))) {
                            bestPeerId = neighbourNode.id;
                        }
                    }

                    neighboursIdsBuffer.clear();
                }

                // zoom out to the highest level
                if (newNode.maxLayer() > entrypointCopy.maxLayer()) {
                    // JK: this is thread safe because we get the global lock when we add a level
                    this.entryPoint = newNode;
                }
            }
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }

    @Override
    public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {

        TravelingCosts<Integer, TDistance> destinationTravelingCosts = new TravelingCosts<>(
                (x, y) -> this.distanceFunction.distance(destination, items.get(x).getVector())
        , -1);

        MutableIntList resultIds = new IntArrayList(k + 1);

        Node entrypointCopy = entryPoint;

        int bestPeerId = findBestPeer(entrypointCopy.id, entrypointCopy.maxLayer(), 0, destinationTravelingCosts);

        runKnnAtLayer(bestPeerId, destinationTravelingCosts, resultIds, 0, k);

        MutableList<SearchResult<TItem, TDistance>> results = resultIds.collect(id -> {
            TItem item = items.get(id);
            TDistance distance = this.distanceFunction.distance(destination, items.get(id).getVector());
            return new SearchResult<>(item, distance);
        });

        Collections.sort(results);
        return results;
    }

    @Override
    public void save(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    /**
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link HnswIndex#save(File)} method.
     *
     * @param file file to initialize the small world from
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <ID, VECTOR, TItem extends Item<ID, VECTOR>, TDistance extends Comparable<TDistance>> HnswIndex<ID, VECTOR, TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link HnswIndex#save(File)} method.
     *
     * @param inputStream InputStream to initialize the small world from
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <ID, VECTOR, TItem extends Item<ID, VECTOR>, TDistance extends Comparable<TDistance>> HnswIndex<ID, VECTOR, TItem, TDistance> load(InputStream inputStream) throws IOException {
        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (HnswIndex<ID, VECTOR, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    private int findBestPeer(int entrypointId,
                             int fromLayer,
                             int toLayer,
                             TravelingCosts<Integer, TDistance> destinationTravelingCost) {

        int bestPeerId = entrypointId;

        MutableIntList neighboursIdsBuffer = new IntArrayList(1);

        for (int layer = fromLayer; layer > toLayer; layer--) {
            runKnnAtLayer(bestPeerId, destinationTravelingCost, neighboursIdsBuffer, layer, 1);

            int candidateBestPeerId = neighboursIdsBuffer.getFirst();

            neighboursIdsBuffer.clear();

            if (bestPeerId == candidateBestPeerId) {
                break;
            }

            bestPeerId = candidateBestPeerId;
        }
        return bestPeerId;
    }

    /**
     * The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
     * Article: Section 4. Algorithm 2.
     *
     * @param entryPointId The identifier of the entry point for the search.
     * @param targetCosts The traveling costs for the search target.
     * @param resultList The list of identifiers of the nearest neighbours at the level.
     * @param layer The layer to perform search at.
     * @param k The number of the nearest neighbours to get from the layer.
     */
    private void runKnnAtLayer(int entryPointId, TravelingCosts<Integer, TDistance> targetCosts, MutableIntList resultList, int layer, int k) {

        // prepare tools
        Comparator<Integer> closerIsOnTop = targetCosts.reversed();

        // prepare collections

        MutableIntList expansionBuffer = expansionBufferPool.borrowObject();
        VisitedBitSet visitedSet = visitedBitSetPool.borrowObject();

        IntBinaryHeap resultHeap = new IntBinaryHeap(resultList, targetCosts);
        IntBinaryHeap expansionHeap = new IntBinaryHeap(expansionBuffer, closerIsOnTop);

        resultHeap.push(entryPointId);
        expansionHeap.push(entryPointId);
        visitedSet.add(entryPointId);

        // run bfs
        while (!expansionHeap.getBuffer().isEmpty()) {
            // get next candidate to check and expand
            int toExpandId = expansionHeap.pop();
            int farthestResultId = resultHeap.getBuffer().getFirst();
            if (DistanceUtils.gt(targetCosts.from(toExpandId), targetCosts.from(farthestResultId))) {
                // the closest candidate is farther than farthest result
                break;
            }

            // expand candidate

            Node node = nodes.get(toExpandId);
            synchronized (node) {

                IntList neighboursIds = node.connections[layer];

                for (int i = 0; i < neighboursIds.size(); i++) {
                    int neighbourId = neighboursIds.get(i);

                    if (!visitedSet.contains(neighbourId)) {
                        // enqueue perspective neighbours to expansion list
                        farthestResultId = resultHeap.getBuffer().getFirst();
                        if (resultHeap.getBuffer().size() < k
                                || DistanceUtils.lt(targetCosts.from(neighbourId), targetCosts.from(farthestResultId))) {
                            expansionHeap.push(neighbourId);
                            resultHeap.push(neighbourId);
                            if (resultHeap.getBuffer().size() > k) {
                                resultHeap.pop();
                            }
                        }

                        // update visited list
                        visitedSet.add(neighbourId);
                    }
                }
            }
        }

        visitedSet.clear();
        visitedBitSetPool.returnObject(visitedSet);

        expansionBuffer.clear();
        expansionBufferPool.returnObject(expansionBuffer);
    }

    /**
     * Gets the distance between 2 items.
     *
     * @param fromId The identifier of the "from" item.
     * @param toId The identifier of the "to" item.
     * @return The distance between items.
     */
    private TDistance calculateDistance(int fromId, int toId) {

        TItem fromItem = items.get(fromId);
        TItem toItem = items.get(toId);

        return this.distanceFunction.distance(fromItem.getVector(), toItem.getVector());
    }

    /**
     * Gets the random layer.
     *
     * @param generator The random numbers generator.
     * @param lambda Poisson lambda.
     * @return The layer value.
     */
    private int randomLayer(DotNetRandom generator, double lambda) {
        double r = -Math.log(generator.nextDouble()) * lambda;
        return (int)r;
    }

    /**
     * The implementation of the node in hnsw graph.
     */
    static class Node implements Serializable {

        private int id;

        private MutableIntList[] connections;

        /**
         * Gets the max layer where the node is presented.
         */
        int maxLayer() {
            return this.connections.length - 1;
        }
    }

    /**
     * The abstract class representing algorithm to control node capacity.
     */
    abstract class Algorithm implements Serializable {

        /**
         * Creates a new instance of the {@link Node} struct. Controls the exact type of connection lists.
         *
         * @param nodeId The identifier of the node.
         * @param maxLayer The max layer where the node is presented.
         * @return The new instance.
         */
        Node newNode(int nodeId, int maxLayer) {
            IntArrayList[] connections = new IntArrayList[maxLayer + 1];

            for (int layer = 0; layer <= maxLayer; layer++) {
                // M + 1 neighbours to not realloc in addConnection when the level is full
                int layerM = this.getM(layer) + 1;
                connections[layer] = new IntArrayList(layerM);
            }

            Node node = new Node();
            node.id = nodeId;
            node.connections = connections;

            return node;
        }

        /**
         * The algorithm which selects best neighbours from the candidates for the given node.
         *
         * @param candidatesIds The identifiers of candidates to neighbourhood.
         * @param travelingCosts Traveling costs to compare candidates.
         * @param layer The layer of the neighbourhood.
         * @return Best nodes selected from the candidates.
         */
        abstract MutableIntList selectBestForConnecting(MutableIntList candidatesIds,
                                                        TravelingCosts<Integer, TDistance> travelingCosts,
                                                        int layer);
        /**
         * Get maximum allowed connections for the given level.
         *
         * Article: Section 4.1:
         * "Selection of the Mmax0 (the maximum number of connections that an element can have in the zero layer) also
         * has a strong influence on the search performance, especially in case of high quality(high recall) search.
         * Simulations show that setting Mmax0 to M(this corresponds to kNN graphs on each layer if the neighbors
         * selection heuristic is not used) leads to a very strong performance penalty at high recall.
         * Simulations also suggest that 2∙M is a good choice for Mmax0;
         * setting the parameter higher leads to performance degradation and excessive memory usage."
         *
         * @param layer The level of the layer.
         *
         * @return The maximum number of connections.
         */
        int getM(int layer) {
            return layer == 0 ? 2 * HnswIndex.this.m : HnswIndex.this.m;
        }

        /**
         * Tries to connect the node with the new neighbour.
         *
         * @param node The node to add neighbour to.
         * @param neighbour The new neighbour.
         * @param layer The layer to add neighbour to.
         */
        void connect(Node node, Node neighbour, int layer) {

            node.connections[layer].add(neighbour.id);
            if (node.connections[layer].size() > this.getM(layer)) {
                TravelingCosts<Integer, TDistance> travelingCosts = new TravelingCosts<>(HnswIndex.this::calculateDistance, node.id);
                node.connections[layer] = this.selectBestForConnecting(node.connections[layer], travelingCosts, layer);
            }
        }
    }

    /**
     * The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
     * Article: Section 4. Algorithm 3.
     */
    class Algorithm3 extends Algorithm {

        /**
         * {@inheritDoc}
         */
        @Override
        MutableIntList selectBestForConnecting(MutableIntList candidatesIds, TravelingCosts<Integer, TDistance> travelingCosts, int layer) {
            /*
             * q ← this
             * return M nearest elements from C to q
             */

            // !NO COPY! in-place selection
            int bestN = this.getM(layer);
            IntBinaryHeap candidatesHeap = new IntBinaryHeap(candidatesIds, travelingCosts);
            while (candidatesHeap.getBuffer().size() > bestN) {
                candidatesHeap.pop();
            }

            return candidatesHeap.getBuffer();
        }
    }

    /**
     * The implementation of the SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections) algorithm.
     * Article: Section 4. Algorithm 4.
     */
    class Algorithm4 extends Algorithm {

        /**
         * {@inheritDoc}
         */
        @Override
        MutableIntList selectBestForConnecting(MutableIntList candidatesIds, TravelingCosts<Integer, TDistance> travelingCosts, int layer) {

            Comparator<Integer> closerIsOnTop = travelingCosts.reversed();

            int layerM = this.getM(layer);

            IntBinaryHeap resultHeap = new IntBinaryHeap(new IntArrayList(layerM + 1), travelingCosts);
            IntBinaryHeap candidatesHeap = new IntBinaryHeap(candidatesIds, closerIsOnTop);

            // expand candidates option is enabled
            MutableIntList candidatesHeapBuffer = candidatesHeap.getBuffer();
            if (expandBestSelection) {

                MutableIntSet visited = IntHashSet.newSet(candidatesHeapBuffer);

                for (int i = 0; i < candidatesHeapBuffer.size(); i++) {

                    int candidateId = candidatesHeapBuffer.get(i);

                    Node candidateNode = nodes.get(candidateId);
                    synchronized (candidateNode) {
                        MutableIntList candidateNodeConnections = candidateNode.connections[layer];

                        for (int j = 0; j < candidateNodeConnections.size(); j++) {
                            int candidateNeighbourId = candidateNodeConnections.get(j);

                            if (!visited.contains(candidateNeighbourId)) {
                                candidatesHeap.push(candidateNeighbourId);
                                visited.add(candidateNeighbourId);
                            }
                        }
                    }
                }
            }

            // main stage of moving candidates to result
            IntBinaryHeap discardedHeap = new IntBinaryHeap(new IntArrayList(candidatesHeapBuffer.size()), closerIsOnTop);

            MutableIntList resultHeapBuffer = resultHeap.getBuffer();

            while (!candidatesHeapBuffer.isEmpty() && resultHeapBuffer.size() < layerM) {
                int candidateId = candidatesHeap.pop();

                int farestResultId = resultHeapBuffer.isEmpty() ? 0 : resultHeapBuffer.getFirst();

                if (resultHeapBuffer.isEmpty()
                        || DistanceUtils.lt(travelingCosts.from(candidateId), travelingCosts.from(farestResultId))) {
                    resultHeap.push(candidateId);

                }  else if (keepPrunedConnections) {
                    discardedHeap.push(candidateId);
                }
            }

            // keep pruned option is enabled
            if (keepPrunedConnections) {
                while (!discardedHeap.getBuffer().isEmpty() && resultHeapBuffer.size() < layerM) {
                    resultHeap.push(discardedHeap.pop());
                }
            }

            return resultHeapBuffer;
        }
    }

    public static class Builder <TVector, TDistance extends Comparable<TDistance>> {

        private DistanceFunction<TVector, TDistance> distanceFunction;
        private int maxItemCount;

        private int m = 10;
        private double levelLambda = 1 / Math.log(this.m);
        private NeighbourSelectionHeuristic neighbourHeuristic = NeighbourSelectionHeuristic.SELECT_SIMPLE;
        private int constructionPruning = 200;
        private boolean expandBestSelection = false;
        private boolean keepPrunedConnections = true;

        private int randomSeed = (int) System.currentTimeMillis();

        public Builder(DistanceFunction<TVector, TDistance> distanceFunction, int maxItemCount) {
            this.distanceFunction = distanceFunction;
            this.maxItemCount = maxItemCount;
        }

        public Builder<TVector, TDistance> setM(int m) {
            this.m = m;
            return this;
        }

        public Builder<TVector, TDistance> setLevelLambda(double levelLambda) {
            this.levelLambda = levelLambda;
            return this;
        }

        public Builder<TVector, TDistance> setNeighbourHeuristic(NeighbourSelectionHeuristic neighbourHeuristic) {
            this.neighbourHeuristic = neighbourHeuristic;
            return this;
        }

        public Builder<TVector, TDistance> setConstructionPruning(int constructionPruning) {
            this.constructionPruning = constructionPruning;
            return this;
        }

        public Builder<TVector, TDistance> setExpandBestSelection(boolean expandBestSelection) {
            this.expandBestSelection = expandBestSelection;
            return this;
        }

        public Builder<TVector, TDistance> setKeepPrunedConnections(boolean keepPrunedConnections) {
            this.keepPrunedConnections = keepPrunedConnections;
            return this;
        }

        public Builder<TVector, TDistance> setRandomSeed(int randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }

        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<TId, TVector, TItem, TDistance>(this);
        }

    }

}

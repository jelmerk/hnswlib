package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * The implementation of a hierarchical small world graph.
 *
 * @param <TItem> The type of items to connect into small world.
 */
class Graph<TItem> implements Serializable {

    // The distance.
    private DistanceFunction<TItem> distance;

    // The core.
    private Core core;

    // The entry point.
    private Node entryPoint;

    // The knn searcher.
    private Searcher searcher;

    // The parameters of the world
    private SmallWorld.Parameters parameters;

    /**
     * Initializes a new instance of the {@link Graph} class.
     *
     * @param distance The distance function.
     * @param parameters The parameters of the world.
     */
    Graph(DistanceFunction<TItem> distance, SmallWorld.Parameters parameters) {
        this.distance = distance;
        this.parameters = parameters;
    }

    /**
     * Gets the parameters.
     */
    SmallWorld.Parameters getParameters() {
        return parameters;
    }

    /**
     * Creates graph from the given items.
     * Contains implementation of INSERT(hnsw, q, M, Mmax, efConstruction, mL) algorithm.
     * Article: Section 4. Algorithm 1.
     *
     * @param items The items to insert.
     * @param generator The random number generator to distribute nodes across layers.
     */
    void build(List<TItem> items, DotNetRandom generator) {
        if (items == null || items.isEmpty()) {
            return;
        }

        Core core = new Core(this.distance, this.getParameters(), items);
        core.allocateNodes(generator);

        Node entryPoint = core.nodes.get(0);
        Searcher searcher = new Searcher(core);
        DistanceFunction<Integer> nodeDistance = core::calculateDistance;
        List<Integer> neighboursIdsBuffer = new ArrayList<>(core.getAlgorithm().getM(0) + 1);

        for (int nodeId = 1; nodeId < core.getNodes().size(); nodeId++) {
            /*
             * W ← ∅ // list for the currently found nearest elements
             * ep ← get enter point for hnsw
             * L ← level of ep // top layer for hnsw
             * l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
             * for lc ← L … l+1
             *   W ← SEARCH-LAYER(q, ep, ef=1, lc)
             *   ep ← get the nearest element from W to q
             * for lc ← min(L, l) … 0
             *   W ← SEARCH-LAYER(q, ep, efConstruction, lc)
             *   neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4
             *     for each e ∈ neighbors // shrink connections if needed
             *       eConn ← neighbourhood(e) at layer lc
             *       if │eConn│ > Mmax // shrink connections of e if lc = 0 then Mmax = Mmax0
             *         eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc) // alg. 3 or alg. 4
             *         set neighbourhood(e) at layer lc to eNewConn
             *   ep ← W
             * if l > L
             *   set enter point for hnsw to q
             */

            // zoom in and find the best peer on the same level as newNode
            Node bestPeer = entryPoint;
            Node currentNode = core.getNodes().get(nodeId);
            TravelingCosts<Integer> currentNodeTravelingCosts = new TravelingCosts<>(nodeDistance, nodeId);
            for (int layer = bestPeer.getMaxLayer(); layer > currentNode.getMaxLayer(); layer--) {
                searcher.runKnnAtLayer(bestPeer.getId(), currentNodeTravelingCosts, neighboursIdsBuffer, layer, 1);
                bestPeer = core.getNodes().get(neighboursIdsBuffer.get(0));
                neighboursIdsBuffer.clear();
            }

            // connecting new node to the small world
            for (int layer = Math.min(currentNode.getMaxLayer(), entryPoint.getMaxLayer()); layer >= 0; layer--) {
                searcher.runKnnAtLayer(bestPeer.getId(), currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.getParameters().getConstructionPruning());
                List<Integer> bestNeighboursIds = core.getAlgorithm().selectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

                for (int newNeighbourId : bestNeighboursIds) {
                    core.getAlgorithm().connect(currentNode, core.getNodes().get(newNeighbourId), layer);
                    core.getAlgorithm().connect(core.getNodes().get(newNeighbourId), currentNode, layer);

                    // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                    if (currentNodeTravelingCosts.from(newNeighbourId) < currentNodeTravelingCosts.from(bestPeer.getId())) {
                        bestPeer = core.getNodes().get(newNeighbourId);
                    }
                }

                neighboursIdsBuffer.clear();
            }

            // zoom out to the highest level
            if (currentNode.getMaxLayer() > entryPoint.getMaxLayer()) {
                entryPoint = currentNode;
            }
        }

        // construction is done
        this.core = core;
        this.entryPoint = entryPoint;
        this.searcher = searcher;
    }

    /**
     * Get k nearest items for a given one.
     * Contains implementation of K-NN-SEARCH(hnsw, q, K, ef) algorithm.
     * Article: Section 4. Algorithm 5.
     *
     * @param destination The given node to get the nearest neighbourhood for.
     * @param k The size of the neighbourhood.
     * @return The list of the nearest neighbours.
     */
    List<SmallWorld.KNNSearchResult<TItem>> kNearest(TItem destination, int k) {

        DistanceFunction<Integer>  runtimeDistance = (x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distance.distance(destination, this.core.getItems().get(nodeId));
        };

        Node bestPeer = this.entryPoint;
        // TODO: hack we know that destination id is -1.

        TravelingCosts<Integer> destinationTravelingCosts = new TravelingCosts<>((x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distance.distance(destination, this.core.getItems().get(nodeId));
        }, -1);

        List<Integer> resultIds = new ArrayList<>(k + 1);

        for (int layer = this.entryPoint.getMaxLayer(); layer > 0; layer--) {
            this.searcher.runKnnAtLayer(bestPeer.getId(), destinationTravelingCosts, resultIds, layer, 1);
            bestPeer = this.core.getNodes().get(resultIds.get(0));
            resultIds.clear();
        }

        this.searcher.runKnnAtLayer(bestPeer.getId(), destinationTravelingCosts, resultIds, 0, k);

        return resultIds.stream()
                .map(id -> {
                    SmallWorld.KNNSearchResult<TItem> result = new SmallWorld.KNNSearchResult<>();
                    result.setId(id);
                    result.setItem(this.core.getItems().get(id));
                    result.setDistance(runtimeDistance.distance(id,  -1));
                    return result;
                })
                .collect(Collectors.toList());
    }

    /**
     * Prints edges of the graph.
     *
     * @return String representation of the graph's edges,
     */
    String print() {
        StringBuilder buffer = new StringBuilder();
        for (int layer = this.entryPoint.getMaxLayer(); layer >= 0; --layer) {
            buffer.append(String.format("[LEVEL %s]%n", layer));
            int finalLevel = layer;

            bfs(this.core, this.entryPoint, layer, node -> {
                String neighbours = node.getConnections(finalLevel).stream().map(String::valueOf)
                        .collect(Collectors.joining(","));
                buffer.append(String.format("(%d) -> {%s}%n", node.getId(), neighbours));
            });
            buffer.append(String.format("%n"));
        }

        return buffer.toString();
    }

    /**
     * Runs breadth first search.
     *
     * @param core The graph core.
     * @param entryPoint The entry point.
     * @param layer The layer of the graph where to run BFS.
     * @param visitConsumer The action to perform on each node.
     */
    void bfs(Core core, Node entryPoint, int layer, Consumer<Node> visitConsumer) {

        Set<Integer> visitedIds = new HashSet<>();
        Queue<Integer> expansionQueue = new LinkedList<>(Collections.singleton(entryPoint.getId()));

        while (!expansionQueue.isEmpty()) {

            Node currentNode = core.getNodes().get(expansionQueue.remove());
            if (!visitedIds.contains(currentNode.getId())) {
                visitConsumer.accept(currentNode);
                visitedIds.add(currentNode.getId());
                expansionQueue.addAll(currentNode.getConnections(layer));
            }
        }
    }


    /**
     * The graph core.
     */
    class Core implements Serializable {

        private static final float MISSING_CACHE_VALUE_MARKER = Float.MIN_VALUE;

        // The original distance function.
        private DistanceFunction<TItem> distance;

        // The distance cache.
        private DistanceCache distanceCache;

        // The parameters of the world.
        private SmallWorld.Parameters parameters;

        // The original items.
        private List<TItem> items;

        // The graph nodes
        private List<Node> nodes;

        // Algorithm for allocating and managing nodes capacity
        private Node.Algorithm<TItem> algorithm;

        /**
         * Initializes a new instance of the {@link Core} class.
         *
         * @param distance The distance function in the items space.
         * @param parameters The parameters of the world.
         * @param items The original items.
         */
        Core(DistanceFunction<TItem> distance, SmallWorld.Parameters parameters, List<TItem> items) {
            this.distance = distance;
            this.parameters = parameters;
            this.items = items;

            switch (this.parameters.getNeighbourHeuristic()) {
                case SELECT_SIMPLE:
                    this.algorithm = new Node.Algorithm3<>(this);
                    break;
                case SELECT_HEURISTIC:
                    this.algorithm = new Node.Algorithm4<>(this);
                    break;
            }

            if (this.parameters.isEnableDistanceCacheForConstruction()) {
                this.distanceCache = new DistanceCache(this.items.size());
            }
        }

        /**
         * Gets the graph nodes corresponding to {@link Core#getItems()}
         */
        List<Node> getNodes() {
            return Collections.unmodifiableList(nodes);
        }

        /**
         * Gets the items associated with the {@link Core#getNodes()}
         */
        List<TItem> getItems() {
            return Collections.unmodifiableList(items);
        }

        /**
         * Gets the algorithm for allocating and managing nodes capacity.
         */
        Node.Algorithm<TItem> getAlgorithm() {
            return algorithm;
        }

        /**
         * Gets parameters of the small world.
         */
        SmallWorld.Parameters getParameters() {
            return parameters;
        }

        /**
         * Gets the distance function
         */
        DistanceFunction<TItem> getDistance() {
            return distance;
        }

        /**
         * Initializes node array for building graph.
         *
         * @param generator The random number generator to assign layers.
         */
        void allocateNodes(DotNetRandom generator) {
            List<Node> nodes = new ArrayList<>(this.items.size());
            for (int id = 0; id < this.items.size(); id++) {
                nodes.add(this.getAlgorithm().newNode(id, randomLayer(generator, this.parameters.getLevelLambda())));
            }

            this.nodes = nodes;
        }

        /**
         * Gets the distance between 2 items.
         *
         * @param fromId The identifier of the "from" item.
         * @param toId The identifier of the "to" item.
         * @return The distance between items.
         */
        float calculateDistance(int fromId, int toId) {
            float result = MISSING_CACHE_VALUE_MARKER;
            if (distanceCache != null) {
                result = this.distanceCache.getValueOrDefault(fromId, toId, MISSING_CACHE_VALUE_MARKER);
            }

            if (result == MISSING_CACHE_VALUE_MARKER) {
                result = this.distance.distance(this.getItems().get(fromId), this.getItems().get(toId));
                if (this.distanceCache != null) {
                    this.distanceCache.setValue(fromId, toId, result);
                }
            }
            return result;
        }

        /**
         * Gets the random layer.
         *
         * @param generator The random numbers generator.
         * @param lambda Poisson lambda.
         * @return The layer value.
         */
        private  int randomLayer(DotNetRandom generator, double lambda) {
            double r = -Math.log(generator.nextDouble()) * lambda;
            return (int) r;
        }
    }

    /**
     * Bitset for tracking visited nodes.
     */
    class VisitedBitSet implements Serializable {

        private int[] buffer;

        /**
         * Initializes a new instance of the {@link VisitedBitSet} class.
         *
         * @param nodesCount The number of nodes to track in the set.
         */
        VisitedBitSet(int nodesCount) {
            this.buffer = new int[(nodesCount >> 5) + 1];
        }

        /**
         * Checks whether the node is already in the set.
         *
         * @param nodeId The identifier of the node.
         * @return True if the node is in the set.
         */
        boolean contains(int nodeId) {
            int carrier = this.buffer[nodeId >> 5];
            return ((1 << (nodeId & 31)) & carrier) != 0;
        }

        /**
         * Adds the node id to the set.
         *
         * @param nodeId The node id to add
         */
        void add(int nodeId)  {
            int mask = 1 << (nodeId & 31);
            this.buffer[nodeId >> 5] |= mask;
        }

        /**
         * Clears the set.
         */
        void clear() {
            Arrays.fill(this.buffer, 0);
        }
    }

    /**
     * The graph searcher.
     */
    class Searcher implements Serializable {
        private Core core;
        private List<Integer> expansionBuffer;
        private VisitedBitSet visitedSet;

        /**
         * Initializes a new instance of the {@link Searcher} class.
         *
         * @param core The core of the graph.
         */
        Searcher(Core core) {
            this.core = core;
            this.expansionBuffer = new ArrayList<>();
            this.visitedSet = new VisitedBitSet(core.getNodes().size());
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
        void runKnnAtLayer(int entryPointId, TravelingCosts<Integer> targetCosts, List<Integer> resultList, int layer, int k) {
            /*
             * v ← ep // set of visited elements
             * C ← ep // set of candidates
             * W ← ep // dynamic list of found nearest neighbors
             * while │C│ > 0
             *   c ← extract nearest element from C to q
             *   f ← get furthest element from W to q
             *   if distance(c, q) > distance(f, q)
             *     break // all elements in W are evaluated
             *   for each e ∈ neighbourhood(c) at layer lc // update C and W
             *     if e ∉ v
             *       v ← v ⋃ e
             *       f ← get furthest element from W to q
             *       if distance(e, q) < distance(f, q) or │W│ < ef
             *         C ← C ⋃ e
             *         W ← W ⋃ e
             *         if │W│ > ef
             *           remove furthest element from W to q
             * return W
             */

            // prepare tools
            Comparator<Integer> fartherIsOnTop = targetCosts;
            Comparator<Integer> closerIsOnTop = fartherIsOnTop.reversed();

            // prepare collections
            // TODO: Optimize by providing buffers
            BinaryHeap<Integer> resultHeap = new BinaryHeap<>(resultList, fartherIsOnTop);
            BinaryHeap<Integer> expansionHeap = new BinaryHeap<>(this.expansionBuffer, closerIsOnTop);

            resultHeap.push(entryPointId);
            expansionHeap.push(entryPointId);
            this.visitedSet.add(entryPointId);

            // run bfs
            while (expansionHeap.getBuffer().size() > 0) {
                // get next candidate to check and expand
                Integer toExpandId = expansionHeap.pop();
                Integer farthestResultId = resultHeap.getBuffer().get(0);
                if (targetCosts.from(toExpandId) > targetCosts.from(farthestResultId)) {
                    // the closest candidate is farther than farthest result
                    break;
                }

                // expand candidate
                List<Integer> neighboursIds = this.core.getNodes().get(toExpandId).getConnections(layer);
                for (Integer neighbourId : neighboursIds) {
                    if (!this.visitedSet.contains(neighbourId)) {
                        // enqueue perspective neighbours to expansion list
                        farthestResultId = resultHeap.getBuffer().get(0);
                        if (resultHeap.getBuffer().size() < k
                                || targetCosts.from(neighbourId) < targetCosts.from(farthestResultId)) {
                            expansionHeap.push(neighbourId);
                            resultHeap.push(neighbourId);
                            if (resultHeap.getBuffer().size() > k) {
                                resultHeap.pop();
                            }
                        }

                        // update visited list
                        this.visitedSet.add(neighbourId);
                    }
                }
            }

            this.expansionBuffer.clear();
            this.visitedSet.clear();
        }

    }
}

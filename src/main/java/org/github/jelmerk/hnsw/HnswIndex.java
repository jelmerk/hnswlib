package org.github.jelmerk.hnsw;


import org.github.jelmerk.Index;
import org.github.jelmerk.SearchResult;

import java.io.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class HnswIndex<TItem, TDistance extends Comparable<TDistance>>
        implements Index<TItem, TDistance>, Serializable {

    private final DotNetRandom random;
    private final Parameters parameters;
    private final DistanceFunction<TItem, TDistance> distanceFunction;

    private final List<TItem> items;
    private final List<NodeNew> nodes;
    private final AlgorithmNew algorithm;

    private NodeNew entryPoint = null;
    private int maxLevel = -1;

    private ReentrantLock globalLock;

    public HnswIndex(Parameters parameters,
                     DistanceFunction<TItem, TDistance> distanceFunction) {

        this(new DotNetRandom(), parameters, distanceFunction);
    }

    public HnswIndex(DotNetRandom random,
                     Parameters parameters,
                     DistanceFunction<TItem, TDistance> distanceFunction) {

        this.random = random;
        this.parameters = parameters;
        this.distanceFunction = distanceFunction;


        if (this.parameters.getNeighbourHeuristic() == NeighbourSelectionHeuristic.SELECT_SIMPLE) {
            this.algorithm = new Algorithm3New();
        } else {
            this.algorithm = new Algorithm4New();
        }

        this.globalLock = new ReentrantLock();

        this.items = Collections.synchronizedList(new ArrayList<>());
        this.nodes = Collections.synchronizedList(new ArrayList<>());
    }

    @Override
    public int add(TItem item) {

        int id;
        synchronized (items) {
            items.add(item);
            id = items.size() - 1;
        }

        NodeNew newNode = this.algorithm.newNode(id, randomLayer(random, this.parameters.getLevelLambda()));
        nodes.add(newNode);

        // TODO i guess we need to sync this somehow too
        if (this.entryPoint == null) {
            this.entryPoint = newNode;
        }

        int maxlevelcopy = maxLevel;

        if (newNode.getMaxLayer() > maxlevelcopy) {
            globalLock.lock();
        }

        try {

            // zoom in and find the best peer on the same level as newNode
            NodeNew bestPeer = this.entryPoint;
            NodeNew currentNode = newNode;


            List<Integer> neighboursIdsBuffer = new ArrayList<>(algorithm.getM(0) + 1);


//            /*
//             * W ← ∅ // list for the currently found nearest elements
//             * ep ← get enter point for hnsw
//             * L ← level of ep // top layer for hnsw
//             * l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
//             * for lc ← L … l+1
//             *   W ← SEARCH-LAYER(q, ep, ef=1, lc)
//             *   ep ← get the nearest element from W to q
//             * for lc ← min(L, l) … 0
//             *   W ← SEARCH-LAYER(q, ep, efConstruction, lc)
//             *   neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4
//             *     for each e ∈ neighbors // shrink connections if needed
//             *       eConn ← neighbourhood(e) at layer lc
//             *       if │eConn│ > Mmax // shrink connections of e if lc = 0 then Mmax = Mmax0
//             *         eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc) // alg. 3 or alg. 4
//             *         set neighbourhood(e) at layer lc to eNewConn
//             *   ep ← W
//             * if l > L
//             *   set enter point for hnsw to q
//             */

            // zoom in and find the best peer on the same level as newNode
            TravelingCosts<Integer, TDistance> currentNodeTravelingCosts = new TravelingCosts<>(this::calculateDistance, newNode.getId());
            for (int layer = bestPeer.getMaxLayer(); layer > currentNode.getMaxLayer(); layer--) {
                runKnnAtLayer(bestPeer.getId(), currentNodeTravelingCosts, neighboursIdsBuffer, layer, 1);

                bestPeer = nodes.get(neighboursIdsBuffer.get(0));
                neighboursIdsBuffer.clear();
            }


            // connecting new node to the small world
            for (int layer = Math.min(currentNode.getMaxLayer(), entryPoint.getMaxLayer()); layer >= 0; layer--) {
                runKnnAtLayer(bestPeer.getId(), currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.parameters.getConstructionPruning());
                List<Integer> bestNeighboursIds = algorithm.selectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

                for (int newNeighbourId : bestNeighboursIds) {

                    NodeNew neighbourNode = nodes.get(newNeighbourId);

                    algorithm.connect(currentNode, neighbourNode, layer);
                    algorithm.connect(neighbourNode, currentNode, layer);

                    // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                    if (DistanceUtils.lt(currentNodeTravelingCosts.from(newNeighbourId), currentNodeTravelingCosts.from(bestPeer.getId()))) {
                        bestPeer = neighbourNode;
                    }
                }

                neighboursIdsBuffer.clear();
            }

            // zoom out to the highest level
            if (currentNode.getMaxLayer() > entryPoint.getMaxLayer()) {
                this.entryPoint = currentNode;
                this.maxLevel = newNode.getMaxLayer();
            }

            return newNode.getId();
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }

    @Override
    public List<SearchResult<TItem, TDistance>> search(TItem destination, int k) {


        DistanceFunction<Integer, TDistance>  runtimeDistance = (x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distanceFunction.distance(destination, this.items.get(nodeId));
        };

        int bestPeerId = this.entryPoint.getId();
        // TODO: hack we know that destination id is -1.

        TravelingCosts<Integer, TDistance> destinationTravelingCosts = new TravelingCosts<>((x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distanceFunction.distance(destination, this.items.get(nodeId));
        }, -1);

        List<Integer> resultIds = new ArrayList<>(k + 1);

        for (int layer = this.entryPoint.getMaxLayer(); layer > 0; layer--) {
            runKnnAtLayer(bestPeerId, destinationTravelingCosts, resultIds, layer, 1);
            bestPeerId = resultIds.get(0);
            resultIds.clear();
        }

        runKnnAtLayer(bestPeerId, destinationTravelingCosts, resultIds, 0, k);

        return resultIds.stream()
                .map(id -> {
                    TItem item = this.items.get(id);
                    TDistance distance = runtimeDistance.distance(id, -1);
                    return new SearchResult<>(item, distance);
                })
                .collect(Collectors.toList());
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
    public static <TItem, TDistance extends Comparable<TDistance>> HnswIndex<TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link SmallWorld#save(File)} method.
     *
     * @param inputStream InputStream to initialize the small world from
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, decimal, int, ...).
     * @return the Small world restored from a file
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TItem, TDistance extends Comparable<TDistance>> HnswIndex<TItem, TDistance> load(InputStream inputStream) throws IOException {
        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (HnswIndex<TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
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
    void runKnnAtLayer(int entryPointId, TravelingCosts<Integer, TDistance> targetCosts, List<Integer> resultList, int layer, int k) {
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
        Comparator<Integer> closerIsOnTop = targetCosts.reversed();

        // prepare collections

        // TODO these where instance variables that originally got reused in searcher and since this visited bitset creates a giant array this is not great

        List<Integer> expansionBuffer = new ArrayList<>();
        VisitedBitSet visitedSet = new VisitedBitSet(nodes.size());

        // TODO: Optimize by providing buffers
        BinaryHeap<Integer> resultHeap = new BinaryHeap<>(resultList, targetCosts);
        BinaryHeap<Integer> expansionHeap = new BinaryHeap<>(expansionBuffer, closerIsOnTop);

        resultHeap.push(entryPointId);
        expansionHeap.push(entryPointId);
        visitedSet.add(entryPointId);

        // run bfs
        while (expansionHeap.getBuffer().size() > 0) {
            // get next candidate to check and expand
            Integer toExpandId = expansionHeap.pop();
            Integer farthestResultId = resultHeap.getBuffer().get(0);
            if (DistanceUtils.gt(targetCosts.from(toExpandId), targetCosts.from(farthestResultId))) {
                // the closest candidate is farther than farthest result
                break;
            }

            // expand candidate

            NodeNew node = this.nodes.get(toExpandId);

            List<Integer> neighboursIds = node.getConnections(layer);
            for (Integer neighbourId : neighboursIds) {
                if (!visitedSet.contains(neighbourId)) {
                    // enqueue perspective neighbours to expansion list
                    farthestResultId = resultHeap.getBuffer().get(0);
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


    /**
     * Prints edges of the graph.
     *
     * @return String representation of the graph's edges,
     */
    public String print() {
        StringBuilder buffer = new StringBuilder();
        for (int layer = this.entryPoint.getMaxLayer(); layer >= 0; --layer) {
            buffer.append(String.format("[LEVEL %s]%n", layer));
            int finalLevel = layer;

            bfs(this.entryPoint, layer, node -> {
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
     * @param entryPoint The entry point.
     * @param layer The layer of the graph where to run BFS.
     * @param visitConsumer The action to perform on each node.
     */
    private void bfs(NodeNew entryPoint, int layer, Consumer<NodeNew> visitConsumer) {

        Set<Integer> visitedIds = new HashSet<>();
        Queue<Integer> expansionQueue = new LinkedList<>(Collections.singleton(entryPoint.getId()));

        while (!expansionQueue.isEmpty()) {

            NodeNew currentNode = nodes.get(expansionQueue.remove());
            if (!visitedIds.contains(currentNode.getId())) {
                visitConsumer.accept(currentNode);
                visitedIds.add(currentNode.getId());
                expansionQueue.addAll(currentNode.getConnections(layer));
            }
        }
    }

    /**
     * Gets the distance between 2 items.
     *
     * @param fromId The identifier of the "from" item.
     * @param toId The identifier of the "to" item.
     * @return The distance between items.
     */
    TDistance calculateDistance(int fromId, int toId) {

        TItem fromItem = this.items.get(fromId);
        TItem toItem = this.items.get(toId);

        return this.distanceFunction.distance(fromItem, toItem);
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
    class NodeNew implements Serializable {

        private int id;

        private List<List<Integer>> connections;

        /**
         * Gets the identifier of the node.
         */
        public int getId() {
            return id;
        }

        /**
         * Gets the max layer where the node is presented.
         */
        public int getMaxLayer() {
            return this.connections.size() - 1;
        }

        /**
         * Gets connections ids of the node at the given layer.
         *
         * @param layer The layer to get connections at.
         * @return The connections of the node at the given layer.
         */
        public List<Integer> getConnections(int layer) {
            return Collections.unmodifiableList(this.connections.get(layer));
        }


    }


    /**
     * The abstract class representing algorithm to control node capacity.
     */
    abstract class AlgorithmNew implements Serializable {

        /// Cache of the distance function between the nodes.
        DistanceFunction<Integer, TDistance> nodeDistance;

        /**
         * Initializes a new instance of the {@link AlgorithmNew} class
         *
         */
        AlgorithmNew() {
            this.nodeDistance = HnswIndex.this::calculateDistance; // TODO do i really want to reference this method in algorithm like this here ?
        }

        /**
         * Creates a new instance of the {@link Node} struct. Controls the exact type of connection lists.
         *
         * @param nodeId The identifier of the node.
         * @param maxLayer The max layer where the node is presented.
         * @return The new instance.
         */

        // TODO should this be in algorithm ?? since its the same for both
        NodeNew newNode(int nodeId, int maxLayer) {
            List<List<Integer>> connections = new ArrayList<>(maxLayer + 1);
            for (int layer = 0; layer <= maxLayer; layer++) {
                // M + 1 neighbours to not realloc in addConnection when the level is full
                int layerM = this.getM(layer) + 1;
                connections.add(new ArrayList<>(layerM));
            }

            NodeNew node = new NodeNew();
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
        abstract List<Integer> selectBestForConnecting(List<Integer> candidatesIds,
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
            return layer == 0 ? 2 * parameters.getM() : parameters.getM();
        }

        /**
         * Tries to connect the node with the new neighbour.
         *
         * @param node The node to add neighbour to.
         * @param neighbour The new neighbour.
         * @param layer The layer to add neighbour to.
         */
        void connect(NodeNew node, NodeNew neighbour, int layer) {
            node.connections.get(layer).add(neighbour.id);
            if (node.connections.get(layer).size() > this.getM(layer)) {
                TravelingCosts<Integer, TDistance> travelingCosts = new TravelingCosts<>(this.nodeDistance, node.id);
                node.connections.set(layer, this.selectBestForConnecting(node.connections.get(layer), travelingCosts, layer));
            }
        }
    }

    /**
     * The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
     * Article: Section 4. Algorithm 3.
     */
    class Algorithm3New extends AlgorithmNew {

        /**
         * {@inheritDoc}
         */
        @Override
        List<Integer> selectBestForConnecting(List<Integer> candidatesIds, TravelingCosts<Integer, TDistance> travelingCosts, int layer) {
            /*
             * q ← this
             * return M nearest elements from C to q
             */

            // !NO COPY! in-place selection
            int bestN = this.getM(layer);
            BinaryHeap<Integer> candidatesHeap = new BinaryHeap<>(candidatesIds, travelingCosts);
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
    class Algorithm4New extends AlgorithmNew {

        /**
         * {@inheritDoc}
         */
        @Override
        List<Integer> selectBestForConnecting(List<Integer> candidatesIds, TravelingCosts<Integer, TDistance> travelingCosts, int layer) {

            /*
             * q ← this
             * R ← ∅    // result
             * W ← C    // working queue for the candidates
             * if expandCandidates  // expand candidates
             *   for each e ∈ C
             *     for each eadj ∈ neighbourhood(e) at layer lc
             *       if eadj ∉ W
             *         W ← W ⋃ eadj
             *
             * Wd ← ∅ // queue for the discarded candidates
             * while │W│ gt 0 and │R│ lt M
             *   e ← extract nearest element from W to q
             *   if e is closer to q compared to any element from R
             *     R ← R ⋃ e
             *   else
             *     Wd ← Wd ⋃ e
             *
             * if keepPrunedConnections // add some of the discarded connections from Wd
             *   while │Wd│ gt 0 and │R│ lt M
             *   R ← R ⋃ extract nearest element from Wd to q
             *
             * return R
             */

            Comparator<Integer> fartherIsOnTop = travelingCosts;
            Comparator<Integer> closerIsOnTop = fartherIsOnTop.reversed();

            int layerM = this.getM(layer);

            BinaryHeap<Integer> resultHeap = new BinaryHeap<>(new ArrayList<>(layerM + 1), fartherIsOnTop);
            BinaryHeap<Integer> candidatesHeap = new BinaryHeap<>(candidatesIds, closerIsOnTop);

            // expand candidates option is enabled
            if (parameters.isExpandBestSelection()) {

                Set<Integer> visited = new HashSet<>(candidatesHeap.getBuffer());

                for (Integer candidateId: candidatesHeap.getBuffer()) {


                    for(Integer candidateNeighbourId : nodes.get(candidateId).getConnections(layer)) {

                        if (!visited.contains(candidateNeighbourId)) {
                            candidatesHeap.push(candidateNeighbourId);
                            visited.add(candidateNeighbourId);
                        }

                    }
                }
            }

            // main stage of moving candidates to result
            BinaryHeap<Integer> discardedHeap = new BinaryHeap<>(new ArrayList<>(candidatesHeap.getBuffer().size()), closerIsOnTop);
            while (!candidatesHeap.getBuffer().isEmpty() && resultHeap.getBuffer().size() < layerM) {
                Integer candidateId = candidatesHeap.pop();

                Integer farestResultId = resultHeap.getBuffer().stream().findFirst().orElse(0);

                if (resultHeap.getBuffer().isEmpty()
                        || DistanceUtils.lt(travelingCosts.from(candidateId), travelingCosts.from(farestResultId))) {
                    resultHeap.push(candidateId);

                }  else if (parameters.isKeepPrunedConnections()) {
                    discardedHeap.push(candidateId);
                }
            }

            // keep pruned option is enabled
            if (parameters.isKeepPrunedConnections()) {
                while (!discardedHeap.getBuffer().isEmpty() && resultHeap.getBuffer().size() < layerM) {
                    resultHeap.push(discardedHeap.pop());
                }
            }

            return resultHeap.getBuffer();
        }
    }

}

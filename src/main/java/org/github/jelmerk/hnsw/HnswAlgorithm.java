package org.github.jelmerk.hnsw;


import org.github.jelmerk.NearestNeighboursAlgorithm;
import org.github.jelmerk.SearchResult;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class HnswAlgorithm<TItem, TDistance extends Comparable<TDistance>>
        implements NearestNeighboursAlgorithm<TItem, TDistance> {

    private final DotNetRandom random;
    private final Parameters parameters;
    private final DistanceFunction<TItem, TDistance> distanceFunction;

    private final List<TItem> items;
    private final List<NodeNew> nodes;
    private final NodeNew.AlgorithmNew<TItem, TDistance> algorithm;

    private NodeNew entryPoint = null;

    public HnswAlgorithm(Parameters parameters,
                         DistanceFunction<TItem, TDistance> distanceFunction) {

        this(new DotNetRandom(), parameters, distanceFunction);
    }

    public HnswAlgorithm(DotNetRandom random,
                         Parameters parameters,
                         DistanceFunction<TItem, TDistance> distanceFunction) {

        this.random = random;
        this.parameters = parameters;
        this.distanceFunction = distanceFunction;


        if (this.parameters.getNeighbourHeuristic() == NeighbourSelectionHeuristic.SELECT_SIMPLE) {
            this.algorithm = new NodeNew.Algorithm3New<TItem, TDistance>(this);
        } else {
            this.algorithm = new NodeNew.Algorithm4New<TItem, TDistance>(this);
        }

        this.items = new ArrayList<>();
        this.nodes = new ArrayList<>();
    }

    @Override
    public TItem getItemById(int id) {
        return this.items.get(id);
    }

    @Override
    public int addItem(TItem item) {

        NodeNew newNode;
        synchronized (items) {
            items.add(item);

            int nodeId = items.size() - 1;

            synchronized (nodes) {
                newNode = this.algorithm.newNode(nodeId, randomLayer(random, this.parameters.getLevelLambda()));
                nodes.add(newNode);
            }
        }


        if (this.entryPoint == null) {
            this.entryPoint = newNode;
        }


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

            bestPeer = nodes.get(neighboursIdsBuffer.get(0)); // todo synchronize
            neighboursIdsBuffer.clear();
        }


        // connecting new node to the small world
        for (int layer = Math.min(currentNode.getMaxLayer(), entryPoint.getMaxLayer()); layer >= 0; layer--) {
            runKnnAtLayer(bestPeer.getId(), currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.parameters.getConstructionPruning());
            List<Integer> bestNeighboursIds = algorithm.selectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

            for (int newNeighbourId : bestNeighboursIds) {

                // TODO this also mutates stuff so we need to sync
                algorithm.connect(currentNode, nodes.get(newNeighbourId), layer); // TODO synchronize
                algorithm.connect(nodes.get(newNeighbourId), currentNode, layer); // TODO synchronize

                // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                if (DistanceUtils.lt(currentNodeTravelingCosts.from(newNeighbourId), currentNodeTravelingCosts.from(bestPeer.getId()))) {
                    bestPeer = nodes.get(newNeighbourId);
                }
            }

            neighboursIdsBuffer.clear();
        }

        // zoom out to the highest level
        if (currentNode.getMaxLayer() > entryPoint.getMaxLayer()) {
            this.entryPoint = currentNode;
        }

        return newNode.getId();
    }

    @Override
    public List<SearchResult<TItem, TDistance>> search(TItem destination, int k) {


        DistanceFunction<Integer, TDistance>  runtimeDistance = (x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distanceFunction.distance(destination, this.items.get(nodeId));
        };

        NodeNew bestPeer = this.entryPoint;
        // TODO: hack we know that destination id is -1.

        TravelingCosts<Integer, TDistance> destinationTravelingCosts = new TravelingCosts<>((x, y) -> {
            int nodeId = x >= 0 ? x : y;
            return this.distanceFunction.distance(destination, this.items.get(nodeId));
        }, -1);

        List<Integer> resultIds = new ArrayList<>(k + 1);

        for (int layer = this.entryPoint.getMaxLayer(); layer > 0; layer--) {
            runKnnAtLayer(bestPeer.getId(), destinationTravelingCosts, resultIds, layer, 1);
            bestPeer = this.nodes.get(resultIds.get(0));
            resultIds.clear();
        }

        runKnnAtLayer(bestPeer.getId(), destinationTravelingCosts, resultIds, 0, k);

        return resultIds.stream()
                .map(id -> {
                    TItem item = this.items.get(id);
                    TDistance distance = runtimeDistance.distance(id, -1);
                    return new SearchResult<>(item, distance);
                })
                .collect(Collectors.toList());
    }

    @Override
    public void saveIndex(OutputStream out) throws IOException {
        try(ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
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
        Comparator<Integer> fartherIsOnTop = targetCosts;
        Comparator<Integer> closerIsOnTop = fartherIsOnTop.reversed();

        // prepare collections

        // TODO these where instance variables that originally got reused in searcher
        List<Integer> expansionBuffer = new ArrayList<>();
        VisitedBitSet visitedSet = new VisitedBitSet(nodes.size()); // TODO synchronize

        // TODO: Optimize by providing buffers
        BinaryHeap<Integer> resultHeap = new BinaryHeap<>(resultList, fartherIsOnTop);
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
            List<Integer> neighboursIds = this.nodes.get(toExpandId).getConnections(layer); // TODO synchronize
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

    NodeNew getNodeById(int id) {
        return nodes.get(id);
    }

    /**
     * Gets parameters of the small world.
     */
    Parameters getParameters() {
        return parameters;
    }

    /**
     * Gets the distance between 2 items.
     *
     * @param fromId The identifier of the "from" item.
     * @param toId The identifier of the "to" item.
     * @return The distance between items.
     */
    TDistance calculateDistance(int fromId, int toId) {

        // TODO make items a synchronized list ??

        TItem fromItem;
        TItem toItem;

        synchronized (items) {
            fromItem = this.items.get(fromId);
            toItem = this.items.get(toId);

        }
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
}

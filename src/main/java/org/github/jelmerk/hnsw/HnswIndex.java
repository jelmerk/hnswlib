package org.github.jelmerk.hnsw;


import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.github.jelmerk.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;

public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private static final long serialVersionUID = 1L;

    private final DotNetRandom random;

    private final DistanceFunction<TVector, TDistance> distanceFunction;

    private final int maxItemCount;
    private final int m;
    private final double levelLambda;
    private final int constructionPruning;

    private final AtomicInteger itemCount;
    private AtomicReferenceArray<TItem> items;
    private AtomicReferenceArray<Node> nodes;

    private final Map<TId, Integer> lookup;

    private volatile Node entryPoint;

    private ReentrantLock globalLock;

    private Pool<VisitedBitSet> visitedBitSetPool;


    private HnswIndex(HnswIndex.Builder<TVector, TDistance> builder) {

        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.m = builder.m;
        this.levelLambda = builder.levelLambda;
        this.constructionPruning = Math.max(builder.constructionPruning, m);


        // TODO JK: do i want to keep supporting heuristic 1 ?

//        if (builder.neighbourHeuristic == NeighbourSelectionHeuristic.SELECT_SIMPLE) {
//            this.algorithm = new Algorithm3();
//        } else {
//            this.algorithm = new Algorithm4();
//        }

        this.random = new DotNetRandom(builder.randomSeed); // TODO JK: get rid of this dot net random and use a ThreadLocalRandom so we don't have to synchronize access

        this.globalLock = new ReentrantLock();

        this.itemCount = new AtomicInteger();
        this.items = new AtomicReferenceArray<>(this.maxItemCount);
        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ConcurrentHashMap<>();

        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount), 12);
    }

    @Override
    public int size() {
        return itemCount.get();
    }

    @Override
    public TItem get(TId id) {
        return items.get(lookup.get(id));
    }

    @Override
    public void add(TItem item) {

        int newNodeId = itemCount.getAndIncrement(); // TODO JK i guess we don't want this count to increase if there's no space, how ?

        if (newNodeId >= this.maxItemCount) {
            throw new IllegalStateException("The number of elements exceeds the specified limit.");
        }

        items.set(newNodeId, item);

        int randomLayer = randomLayer(random, this.levelLambda);

        IntArrayList[] connections = new IntArrayList[randomLayer + 1];

        for (int layer = 0; layer <= randomLayer; layer++) {
            int layerM = randomLayer == 0 ? 2 * this.m : this.m;
            connections[layer] = new IntArrayList(layerM);
        }

        Node newNode = new Node();
        newNode.id = newNodeId;
        newNode.connections = connections;


        nodes.set(newNodeId, newNode);

        lookup.put(item.getId(), newNodeId);

        globalLock.lock();

        Node entrypointCopy = entryPoint;

        if (entryPoint != null && newNode.maxLayer() <= entryPoint.maxLayer()) {
            globalLock.unlock();
        }

        try {
            synchronized (newNode) {

                Node currObj = entrypointCopy;

                if (currObj != null) {

                    if (newNode.maxLayer() < entrypointCopy.maxLayer()) {

                        TDistance curDist = distanceFunction.distance(item.getVector(), items.get(currObj.id).getVector());

                        for (int activeLevel = entrypointCopy.maxLayer(); activeLevel > newNode.maxLayer(); activeLevel--) {

                            boolean changed = true;

                            while (changed) {
                                changed = false;

                                synchronized (currObj) {

//                                    MutableIntList candidateConnections = currObj.connections[activeLevel - 1]; // TODO JK why minus one again ?
                                    MutableIntList candidateConnections = currObj.connections[activeLevel];

                                    for (int i = 0; i < candidateConnections.size(); i++) {

                                        int candidateId = candidateConnections.get(i);

                                        TDistance candidateDistance = distanceFunction.distance(item.getVector(), items.get(candidateId).getVector());
                                        if (DistanceUtils.lt(candidateDistance, curDist)) {
                                            curDist = candidateDistance;
                                            currObj = nodes.get(candidateId);
                                            changed = true;
                                        }
                                    }
                                }

                            }
                        }
                    }


                    for (int level = Math.min(randomLayer, entrypointCopy.maxLayer()); level >= 0; level--) {
                        PriorityQueue<NodeAndDistance<TDistance>> topCandidates =
                                searchBaseLayer(currObj.id, item.getVector(), constructionPruning, level);
                        mutuallyConnectNewElement(item.getVector(), newNodeId, topCandidates, level);
                    }

                }

                // zoom out to the highest level
                if (entryPoint == null || newNode.maxLayer() > entrypointCopy.maxLayer()) {
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


    private void mutuallyConnectNewElement(TVector dataPoint,
                                           int nodeId,
                                           PriorityQueue<NodeAndDistance<TDistance>> topCandidates,
                                           int level) {

        int bestN = level == 0 ? 2 * this.m : this.m;

        MutableIntList nodeConnections = nodes.get(nodeId).connections[level];

        getNeighborsByHeuristic2(topCandidates, m); // this modifies the topCandidates queue
//        getNeighborsByHeuristic1(topCandidates, m); // this modifies the topCandidates queue TODO jk: back to 2

        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId;

            nodeConnections.add(selectedNeighbourId);

            Node neighbourNode = nodes.get(selectedNeighbourId);
            synchronized (neighbourNode) {

                MutableIntList neighbourConnectionsAtLevel = neighbourNode.connections[level];

                if (neighbourConnectionsAtLevel.size() < bestN) {
                    neighbourConnectionsAtLevel.add(nodeId);
                } else {
                    // finding the "weakest" element to replace it with the new one

                    TDistance dMax = distanceFunction.distance(dataPoint, items.get(selectedNeighbourId).getVector());

                    Comparator<NodeAndDistance<TDistance>> comparator = Comparator.<NodeAndDistance<TDistance>>naturalOrder().reversed();

                    PriorityQueue<NodeAndDistance<TDistance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new NodeAndDistance<>(nodeId, dMax));

                    neighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(items.get(selectedNeighbourId).getVector(), items.get(id).getVector());
                        candidates.add(new NodeAndDistance<>(id, dist));
                    });

                    getNeighborsByHeuristic2(candidates, bestN);
//                    getNeighborsByHeuristic1(candidates, bestN); // TODO jk, back to 2

                    neighbourConnectionsAtLevel.clear();

                    while(!candidates.isEmpty()) {
                        neighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }
        }
    }

    // TODO JK: not in the original hnsw impl but i think this is what it should be if you look at the algorithm3 class from the .net impl
    private void getNeighborsByHeuristic1(PriorityQueue<NodeAndDistance<TDistance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<NodeAndDistance<TDistance>> queueClosest = new PriorityQueue<>();

        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while(!queueClosest.isEmpty()) {
            if (topCandidates.size() >= m) {
                break;
            }
            topCandidates.add(queueClosest.poll());
        }

    }

    private void getNeighborsByHeuristic2(PriorityQueue<NodeAndDistance<TDistance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<NodeAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
        List<NodeAndDistance<TDistance>> returnList = new ArrayList<>();

        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            NodeAndDistance<TDistance> currentPair = queueClosest.poll();

            TDistance distToQuery = currentPair.distance;

            boolean good = true;
            for (NodeAndDistance<TDistance> secondPair : returnList) {

                TDistance curdist = distanceFunction.distance(
                    items.get(secondPair.nodeId).getVector(),
                    items.get(currentPair.nodeId).getVector()
                );

                if (DistanceUtils.lt(curdist, distToQuery)) {
                    good = false;
                    break;
                }

            }
            if (good) {
                returnList.add(currentPair);
            }
        }

        topCandidates.addAll(returnList);
    }

    @Override
    public List<SearchResult<TItem, TDistance>>findNearest(TVector destination, int k) {

        Node entrypointCopy = entryPoint;

        Node currObj = entrypointCopy;

        TDistance curDist = distanceFunction.distance(destination, items.get(currObj.id).getVector());

        for (int activeLevel = entrypointCopy.maxLayer(); activeLevel > 0; activeLevel--) {

            boolean changed = true;

            while (changed) {
                changed = false;

                synchronized (currObj) {

//                    MutableIntList candidateConnections = currObj.connections[activeLevel - 1];
                    MutableIntList candidateConnections = currObj.connections[activeLevel];

                    for (int i = 0; i < candidateConnections.size(); i++) {

                        int candidateId = candidateConnections.get(i);

                        TDistance candidateDistance = distanceFunction.distance(destination, items.get(candidateId).getVector());
                        if (DistanceUtils.lt(candidateDistance, curDist)) {
                            curDist = candidateDistance;
                            currObj = nodes.get(candidateId);
                            changed = true;
                        }
                    }
                }

            }
        }

        // TODO JK  The quality of the search is controlled by the ef parameter (corresponding to efConstruction in the construction algorithm).

        // TODO JK in hnswlib they have a parameter called ef thats hardcoded to be 10 and then they do Math.max(ef, k) why ?

        PriorityQueue<NodeAndDistance<TDistance>> topCandidates = searchBaseLayer(
                currObj.id, destination, k, 0);


        // TODO JK this code makes no sense if we have no ef value because the priority queue will never be bigger than k, delete it ? or work out what ef is for

        while(topCandidates.size() > k) {
            topCandidates.poll();
        }

        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            NodeAndDistance<TDistance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(items.get(pair.nodeId), pair.distance));
        }

        return results;
    }

    private PriorityQueue<NodeAndDistance<TDistance>> searchBaseLayer(
            int entryPointId, TVector destination, int k, int layer) {

        VisitedBitSet visitedBitSet = visitedBitSetPool.borrowObject();

        PriorityQueue<NodeAndDistance<TDistance>> topCandidates =
                new PriorityQueue<>(Comparator.<NodeAndDistance<TDistance>>naturalOrder().reversed());
        PriorityQueue<NodeAndDistance<TDistance>> candidateSet = new PriorityQueue<>();

        TDistance distance = distanceFunction.distance(destination, items.get(entryPointId).getVector());

        NodeAndDistance<TDistance> pair = new NodeAndDistance<>(entryPointId, distance);

        topCandidates.add(pair);
        candidateSet.add(pair);
        visitedBitSet.add(entryPointId);

        TDistance lowerBound = distance;

        while(!candidateSet.isEmpty()) {

            NodeAndDistance<TDistance> currentPair = candidateSet.peek();

            if (DistanceUtils.gt(currentPair.distance, lowerBound)) {
                break;
            }

            candidateSet.poll();

            Node node = nodes.get(currentPair.nodeId);

            synchronized (node) {

                MutableIntList candidates = node.connections[layer];

                for (int i = 0; i < candidates.size(); i++) {

                    int candidateId = candidates.get(i);

                    if (!visitedBitSet.contains(candidateId)) {

                        visitedBitSet.add(candidateId);

                        TItem candidate = items.get(candidateId);

                        TDistance candidateDistance = distanceFunction.distance(destination, candidate.getVector());

                        if (DistanceUtils.gt(topCandidates.peek().distance, candidateDistance) || topCandidates.size() < k) {

                            NodeAndDistance<TDistance> candidatePair = new NodeAndDistance<>(candidateId, candidateDistance);

                            candidateSet.add(candidatePair);
                            topCandidates.add(candidatePair);

                            if (topCandidates.size() > k) {
                                topCandidates.poll();
                            }

                            lowerBound = topCandidates.peek().distance;
                        }
                    }
                }

            }
        }

        visitedBitSet.clear();
        visitedBitSetPool.returnObject(visitedBitSet);

        return topCandidates;

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
     * The implementation of the nodeId in hnsw graph.
     */
    static class Node implements Serializable {

        private static final long serialVersionUID = 1L;

        private int id;

        private MutableIntList[] connections;

        /**
         * Gets the max layer where the nodeId is presented.
         */
        int maxLayer() {
            return this.connections.length - 1;
        }
    }

    static class NodeAndDistance<TDistance extends Comparable<TDistance>> implements Comparable<NodeAndDistance<TDistance>> {

        final TDistance distance;
        final int nodeId;

        NodeAndDistance(int nodeId, TDistance distance) {
            this.nodeId = nodeId;
            this.distance = distance;
        }

        @Override
        public int compareTo(NodeAndDistance<TDistance> o) {
            return distance.compareTo(o.distance);
        }

    }

    public static class Builder <TVector, TDistance extends Comparable<TDistance>> {

        private DistanceFunction<TVector, TDistance> distanceFunction;
        private int maxItemCount;

        private int m = 10;
        private double levelLambda = 1 / Math.log(this.m);
        private NeighbourSelectionHeuristic neighbourHeuristic = NeighbourSelectionHeuristic.SELECT_SIMPLE;
        private int constructionPruning = 200;

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

        public Builder<TVector, TDistance> setRandomSeed(int randomSeed) {
            this.randomSeed = randomSeed;
            return this;
        }

        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<>(this);
        }

    }

}

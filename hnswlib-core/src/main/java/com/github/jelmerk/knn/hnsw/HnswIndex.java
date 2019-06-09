package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.*;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.stack.primitive.MutableIntStack;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.stack.mutable.primitive.IntArrayStack;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm
 *
 * @param <TId> type of the external identifier of an item
 * @param <TVector> The type of the vector to perform distance calculation on
 * @param <TItem> The type of items to connect into small world.
 * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
 *
 * @see <a href="https://arxiv.org/abs/1603.09320">
 *     Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {
    // TODO implement Externalizable and store these Nodes more efficiently so the graph loads faster on deserialization

    private static final long serialVersionUID = 1L;

    private final DistanceFunction<TVector, TDistance> distanceFunction;
    private final Comparator<TDistance> distanceComparator;

    private final int maxItemCount;
    private final int m;
    private final int maxM;
    private final int maxM0;
    private final double levelLambda;
    private final int ef;
    private final int efConstruction;
    private final boolean removeEnabled;

    private volatile int itemCount;
    private final AtomicReferenceArray<Node<TItem>> nodes;
    private final MutableIntStack freedIds;

    private final Map<TId, Integer> lookup;

    private volatile Node<TItem> entryPoint;

    private final ReentrantLock globalLock;

    private final ReadWriteLock addRemoveLock;
    private final Lock addLock;
    private final Lock removeLock;

    private final Pool<VisitedBitSet> visitedBitSetPool;

    private HnswIndex(HnswIndex.Builder<TVector, TDistance> builder) {

        this.maxItemCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = builder.distanceComparator;

        this.m = builder.m;
        this.maxM = builder.m;
        this.maxM0 = builder.m * 2;
        this.levelLambda = 1 / Math.log(this.m);
        this.efConstruction = Math.max(builder.efConstruction, m);
        this.ef = builder.ef;
        this.removeEnabled = builder.removeEnabled;

        this.globalLock = new ReentrantLock();

        this.addRemoveLock = new ReentrantReadWriteLock();
        this.addLock = addRemoveLock.readLock();
        this.removeLock = addRemoveLock.writeLock();

        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ConcurrentHashMap<>();

        // TODO jk: how do we determine the pool size just use num processors or what ?
        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());

        this.freedIds = new IntArrayStack();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        synchronized (freedIds) {
            return itemCount - freedIds.size();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Optional<TItem> get(TId id) {
        return Optional.ofNullable(lookup.get(id))
                .flatMap(index -> Optional.ofNullable(nodes.get(index)))
                .map(n -> n.item);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean remove(TId id) {
        if (!removeEnabled) {
            throw new UnsupportedOperationException("Index does not have removes enabled.");
        }

        Integer internalNodeId = lookup.get(id);

        if (id == null) {
            return false;
        } else {
            removeLock.lock();

            try {
                Node<TItem> node = nodes.get(internalNodeId);

                for (int level = node.maxLevel(); level >= 0; level--) {

                    final int finalLevel = level;

                    node.incomingConnections[level].forEach(neighbourId ->
                            nodes.get(neighbourId).outgoingConnections[finalLevel].remove(internalNodeId));

                    node.outgoingConnections[level].forEach(neighbourId ->
                            nodes.get(neighbourId).incomingConnections[finalLevel].remove(internalNodeId));

                }

                // change the entry point to the first outgoing connection at the highest level

                if (entryPoint == node) {
                    for (int level = node.maxLevel(); level >= 0; level--) {

                        MutableIntList outgoingConnections = node.outgoingConnections[level];
                        if (!outgoingConnections.isEmpty()) {
                            entryPoint = nodes.get(outgoingConnections.getFirst());
                            break;
                        }
                    }

                }

                // if we could not change the outgoing connection it means we are the last node

                if (entryPoint == node) {
                    entryPoint = null;
                }

                freedIds.push(internalNodeId);
                return true;

            } finally {
                removeLock.unlock();
            }
        }

        // TODO do we want to do anything to fix up the connections like here https://github.com/andrusha97/online-hnsw/blob/master/include/hnsw/index.hpp#L185
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(TItem item) {
        addLock.lock();

        try {
            int newNodeId;

            synchronized (freedIds) {
                if (freedIds.isEmpty()) {
                    if (itemCount >= this.maxItemCount) {
                        throw new IllegalStateException("The number of elements exceeds the specified limit.");
                    }
                    newNodeId = itemCount++;
                } else {
                    newNodeId = freedIds.pop();
                }
            }

            int randomLevel = assignLevel(item.id(), this.levelLambda);

            IntArrayList[] outgoingConnections = new IntArrayList[randomLevel + 1];

            for (int level = 0; level <= randomLevel; level++) {
                int levelM = randomLevel == 0 ? maxM0 : maxM;
                outgoingConnections[level] = new IntArrayList(levelM);
            }

            IntArrayList[] incomingConnections = removeEnabled ? new IntArrayList[randomLevel + 1] : null;
            if (removeEnabled) {
                for (int level = 0; level <= randomLevel; level++) {
                    int levelM = randomLevel == 0 ? maxM0 : maxM;
                    incomingConnections[level] = new IntArrayList(levelM);
                }
            }

            Node<TItem> newNode = new Node<>(newNodeId, outgoingConnections, incomingConnections, item);

            nodes.set(newNodeId, newNode);

            lookup.put(item.id(), newNodeId);

            globalLock.lock();

            try {

                Node<TItem> entryPointCopy = entryPoint;

                if (entryPoint != null && newNode.maxLevel() <= entryPoint.maxLevel()) {
                    globalLock.unlock();
                }

                synchronized (newNode) {

                    Node<TItem> currObj = entryPointCopy;

                    if (currObj != null) {

                        if (newNode.maxLevel() < entryPointCopy.maxLevel()) {

                            TDistance curDist = distanceFunction.distance(item.vector(), currObj.item.vector());

                            for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > newNode.maxLevel(); activeLevel--) {

                                boolean changed = true;

                                while (changed) {
                                    changed = false;

                                    synchronized (currObj) {
                                        MutableIntList candidateConnections = currObj.outgoingConnections[activeLevel];

                                        for (int i = 0; i < candidateConnections.size(); i++) {

                                            int candidateId = candidateConnections.get(i);

                                            Node<TItem> candidateNode = nodes.get(candidateId);

                                            TDistance candidateDistance = distanceFunction.distance(
                                                    item.vector(),
                                                    candidateNode.item.vector()
                                            );

                                            if (lt(candidateDistance, curDist)) {
                                                curDist = candidateDistance;
                                                currObj = candidateNode;
                                                changed = true;
                                            }
                                        }
                                    }

                                }
                            }
                        }

                        for (int level = Math.min(randomLevel, entryPointCopy.maxLevel()); level >= 0; level--) {
                            PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                                    searchBaseLayer(currObj, item.vector(), efConstruction, level);
                            mutuallyConnectNewElement(newNode, topCandidates, level);
                        }
                    }

                    // zoom out to the highest level
                    if (entryPoint == null || newNode.maxLevel() > entryPointCopy.maxLevel()) {
                        // this is thread safe because we get the global lock when we add a level
                        this.entryPoint = newNode;
                    }
                }
            } finally {
                if (globalLock.isHeldByCurrentThread()) {
                    globalLock.unlock();
                }
            }
        } finally {
            addLock.unlock();
        }
    }


    private void mutuallyConnectNewElement(Node<TItem> newNode,
                                           PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates,
                                           int level) {

        int bestN = level == 0 ? this.maxM0 : this.maxM;

        int newNodeId = newNode.id;
        TVector newItemVector = newNode.item.vector();
        MutableIntList outgoingNewItemConnections = newNode.outgoingConnections[level];

        getNeighborsByHeuristic2(topCandidates, m); // this modifies the topCandidates queue

        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId;

            outgoingNewItemConnections.add(selectedNeighbourId);

            int removedWeakestNodeId = -1; // TODO should i just change this to newNodeId and save one extra condition down.. its not completely correct then though

            Node<TItem> neighbourNode = nodes.get(selectedNeighbourId);
            synchronized (neighbourNode) {

                if (removeEnabled) {
                    neighbourNode.incomingConnections[level].add(newNodeId);
                }

                TVector neighbourVector = neighbourNode.item.vector();

                MutableIntList outgoingNeighbourConnectionsAtLevel = neighbourNode.outgoingConnections[level];

                if (outgoingNeighbourConnectionsAtLevel.size() < bestN) {
                    outgoingNeighbourConnectionsAtLevel.add(newNodeId);
                } else {
                    // finding the "weakest" element to replace it with the new one

                    TDistance dMax = distanceFunction.distance(
                            newItemVector,
                            neighbourNode.item.vector()
                    );

                    Comparator<NodeIdAndDistance<TDistance>> comparator = Comparator
                            .<NodeIdAndDistance<TDistance>>naturalOrder().reversed();

                    PriorityQueue<NodeIdAndDistance<TDistance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new NodeIdAndDistance<>(newNodeId, dMax, distanceComparator));

                    outgoingNeighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(
                                neighbourVector,
                                nodes.get(id).item.vector()
                        );

                        candidates.add(new NodeIdAndDistance<>(id, dist, distanceComparator));
                    });

                    getNeighborsByHeuristic2(candidates, bestN + 1);

                    removedWeakestNodeId = candidates.poll().nodeId;

                    outgoingNeighbourConnectionsAtLevel.clear();

                    while(!candidates.isEmpty()) {
                        outgoingNeighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }

            if (removeEnabled && removedWeakestNodeId != newNodeId && removedWeakestNodeId != -1) {
                Node<TItem> weakestNode = nodes.get(removedWeakestNodeId);
                synchronized (weakestNode) {
                    weakestNode.incomingConnections[level].remove(selectedNeighbourId);
                }
            }
        }
    }

    private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<NodeIdAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
        List<NodeIdAndDistance<TDistance>> returnList = new ArrayList<>();

        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            NodeIdAndDistance<TDistance> currentPair = queueClosest.poll();

            TDistance distToQuery = currentPair.distance;

            boolean good = true;
            for (NodeIdAndDistance<TDistance> secondPair : returnList) {

                TDistance curdist = distanceFunction.distance(
                        nodes.get(secondPair.nodeId).item.vector(),
                        nodes.get(currentPair.nodeId).item.vector()
                );

                if (lt(curdist, distToQuery)) {
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

    /**
     * {@inheritDoc}
     */
    @Override
    public List<SearchResult<TItem, TDistance>>findNearest(TVector destination, int k) {

        if (entryPoint == null) {
            return Collections.emptyList();
        }

        Node<TItem> entryPointCopy = entryPoint;

        Node<TItem> currObj = entryPointCopy;

        TDistance curDist = distanceFunction.distance(destination, currObj.item.vector());

        for (int activeLevel = entryPointCopy.maxLevel(); activeLevel > 0; activeLevel--) {

            boolean changed = true;

            while (changed) {
                changed = false;

                synchronized (currObj) {
                    MutableIntList candidateConnections = currObj.outgoingConnections[activeLevel];

                    for (int i = 0; i < candidateConnections.size(); i++) {

                        int candidateId = candidateConnections.get(i);
//                        Node<TItem> candidateNode = nodes.get(candidateId);

                        TDistance candidateDistance = distanceFunction.distance(
                                destination,
                                nodes.get(candidateId).item.vector()
                        );
                        if (lt(candidateDistance, curDist)) {
                            curDist = candidateDistance;
                            currObj = nodes.get(candidateId);
                            changed = true;
                        }
                    }
                }

            }
        }

        PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates = searchBaseLayer(
                currObj, destination, Math.max(ef, k), 0);

        while(topCandidates.size() > k) {
            topCandidates.poll();
        }

        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            NodeIdAndDistance<TDistance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(nodes.get(pair.nodeId).item, pair.distance, distanceComparator));
        }

        return results;
    }

    private PriorityQueue<NodeIdAndDistance<TDistance>> searchBaseLayer(
            Node<TItem> entryPointNode, TVector destination, int k, int layer) {

        VisitedBitSet visitedBitSet = visitedBitSetPool.borrowObject();

        try {
            PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                    new PriorityQueue<>(Comparator.<NodeIdAndDistance<TDistance>>naturalOrder().reversed());
            PriorityQueue<NodeIdAndDistance<TDistance>> candidateSet = new PriorityQueue<>();

            TDistance distance = distanceFunction.distance(destination, entryPointNode.item.vector());

            NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance, distanceComparator);

            topCandidates.add(pair);
            candidateSet.add(pair);
            visitedBitSet.add(entryPointNode.id);

            TDistance lowerBound = distance;

            while (!candidateSet.isEmpty()) {

                NodeIdAndDistance<TDistance> currentPair = candidateSet.peek();

                if (gt(currentPair.distance, lowerBound)) {
                    break;
                }

                candidateSet.poll();

                Node<TItem> node = nodes.get(currentPair.nodeId);

                synchronized (node) {

                    MutableIntList candidates = node.outgoingConnections[layer];

                    for (int i = 0; i < candidates.size(); i++) {

                        int candidateId = candidates.get(i);

                        if (!visitedBitSet.contains(candidateId)) {

                            visitedBitSet.add(candidateId);

                            TDistance candidateDistance = distanceFunction.distance(destination,
                                    nodes.get(candidateId).item.vector());

                            if (gt(topCandidates.peek().distance, candidateDistance) || topCandidates.size() < k) {

                                NodeIdAndDistance<TDistance> candidatePair =
                                        new NodeIdAndDistance<>(candidateId, candidateDistance, distanceComparator);

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

            return topCandidates;
        } finally {
            visitedBitSet.clear();
            visitedBitSetPool.returnObject(visitedBitSet);
        }
    }

    /**
     * Creates a read only view on top of this index that uses pairwise comparision when doing distance search. And as
     * such can be used as a baseline for assessing the accuracy of the index.
     * Searches will be really slow but give the correct result every time.
     *
     * @return read only view on top of this index that uses pairwise comparision when doing distance search
     */
    public ReadOnlyIndex<TId, TVector, TItem, TDistance> exactView() {
        return new ExactView();
    }

    /**
     * Returns the number of bi-directional links created for every new element during construction.
     *
     * @return the number of bi-directional links created for every new element during construction
     */
    public int getM() {
        return m;
    }

    /**
     * The size of the dynamic list for the nearest neighbors (used during the search)
     *
     * @return The size of the dynamic list for the nearest neighbors
     */
    public int getEf() {
        return ef;
    }

    /**
     * Returns the parameter has the same meaning as ef, but controls the index time / index accuracy.
     *
     * @return the parameter has the same meaning as ef, but controls the index time / index accuracy
     */
    public int getEfConstruction() {
        return efConstruction;
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
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link HnswIndex#save(File)} method.
     *
     * @param file file to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     * @return the index world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance
           > HnswIndex<TId, TVector, TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link HnswIndex#save(Path)} method.
     *
     * @param path path to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     * @return the index world restored from a file
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance
           > HnswIndex<TId, TVector, TItem, TDistance> load(Path path) throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link HnswIndex} instance from a file created by invoking the {@link HnswIndex#save(OutputStream)} method.
     *
     * @param inputStream InputStream to initialize the small world from
     * @param <TId> type of the external identifier of an item
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TItem> The type of items to connect into small world.
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ...).
     * @return the index world restored from a file
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance
           > HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream)
            throws IOException {

        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (HnswIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    public static <TVector, TDistance extends Comparable<TDistance>>
        Builder <TVector, TDistance>
            newBuilder(DistanceFunction<TVector, TDistance> distanceFunction, int maxItemCount) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();
        return new Builder<>(distanceFunction, distanceComparator, maxItemCount);
    }

    public static <TVector, TDistance>
        Builder <TVector, TDistance>
            newBuilder(DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator, int maxItemCount) {

        return new Builder<>(distanceFunction, distanceComparator, maxItemCount);
    }

    private int assignLevel(TId value, double lambda) {

        // by relying on the external id to come up with the level, the graph construction should be a lot mor stable
        // see : https://github.com/nmslib/hnswlib/issues/28

        int hashCode = value.hashCode();

        byte[] bytes = new byte[] {
                (byte) (hashCode >> 24),
                (byte) (hashCode >> 16),
                (byte) (hashCode >> 8),
                (byte) hashCode
        };

        double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE);

        double r = -Math.log(random) * lambda;
        return (int)r;
    }

    private boolean lt(TDistance x, TDistance y) {
        return distanceComparator.compare(x, y) < 0;
    }

    private boolean gt(TDistance x, TDistance y) {
        return distanceComparator.compare(x, y) > 0;
    }

    class ExactView implements ReadOnlyIndex<TId, TVector, TItem, TDistance> {
        @Override
        public int size() {
            return HnswIndex.this.size();
        }

        @Override
        public Optional<TItem> get(TId tId) {
            return HnswIndex.this.get(tId);
        }

        @Override
        public List<SearchResult<TItem, TDistance>> findNearest(TVector vector, int k) {

            Comparator<SearchResult<TItem, TDistance>> comparator = Comparator
                    .<SearchResult<TItem, TDistance>>naturalOrder()
                    .reversed();

            PriorityQueue<SearchResult<TItem, TDistance>> queue = new PriorityQueue<>(k, comparator);

            for (int i = 0; i < itemCount; i++) {
                Node<TItem> node = nodes.get(i);
                if (node == null) {
                    continue;
                }

                TDistance distance = distanceFunction.distance(node.item.vector(), vector);

                SearchResult<TItem, TDistance> searchResult = new SearchResult<>(node.item, distance, distanceComparator);
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
    }

    static class Node<TItem> implements Serializable {

        private static final long serialVersionUID = 1L;

        final int id;

        final MutableIntList[] outgoingConnections;

        final MutableIntList[] incomingConnections;

        final TItem item;

        Node(int id, MutableIntList[] outgoingConnections, MutableIntList[] incomingConnections, TItem item) {
            this.id = id;
            this.outgoingConnections = outgoingConnections;
            this.incomingConnections = incomingConnections;
            this.item = item;
        }

        int maxLevel() {
            return this.outgoingConnections.length - 1;
        }
    }

    static class NodeIdAndDistance<TDistance> implements Comparable<NodeIdAndDistance<TDistance>> {

        final int nodeId;
        final TDistance distance;
        final Comparator<TDistance> distanceComparator;

        NodeIdAndDistance(int nodeId, TDistance distance, Comparator<TDistance> distanceComparator) {
            this.nodeId = nodeId;
            this.distance = distance;
            this.distanceComparator = distanceComparator;
        }

        @Override
        public int compareTo(NodeIdAndDistance<TDistance> o) {
            return  distanceComparator.compare(distance, o.distance);
        }

    }


    /**
     * Builder for initializing an {@link HnswIndex} instance.
     *
     * @param <TVector> The type of the vector to perform distance calculation on
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ..).
     */
    public static class Builder <TVector, TDistance> {

        public static final int DEFAULT_M = 10;
        public static final int DEFAULT_EF = 10;
        public static final int DEFAULT_EF_CONSTRUCTION = 200;
        public static final boolean DEFAULT_REMOVE_ENABLED = false;

        private DistanceFunction<TVector, TDistance> distanceFunction;
        private Comparator<TDistance> distanceComparator;

        private int maxItemCount;

        private int m = DEFAULT_M;
        private int ef = DEFAULT_EF;
        private int efConstruction = DEFAULT_EF_CONSTRUCTION;
        private boolean removeEnabled = DEFAULT_REMOVE_ENABLED;

        /**
         * Constructs a new {@link Builder} instance.
         *
         * @param distanceFunction the distance function
         * @param maxItemCount the maximum number of elements in the index
         */
        Builder(DistanceFunction<TVector, TDistance> distanceFunction,
                Comparator<TDistance> distanceComparator,
                int maxItemCount) {
            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
            this.maxItemCount = maxItemCount;
        }

        /**
         * Sets the number of bi-directional links created for every new element during construction. Reasonable range
         * for m is 2-100. Higher m work better on datasets with high intrinsic dimensionality and/or high recall,
         * while low m work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter
         * also determines the algorithm's memory consumption.
         * As an example for d = 4 random vectors optimal m for search is somewhere around 6, while for high dimensional
         * datasets (word embeddings, good face descriptors), higher M are required (e.g. m = 48, 64) for optimal
         * performance at high recall. The range mM = 12-48 is ok for the most of the use cases. When m is changed one
         * has to update the other parameters. Nonetheless, ef and efConstruction parameters can be roughly estimated by
         * assuming that m * efConstruction is a constant.
         *
         * @param m the number of bi-directional links created for every new element during construction
         * @return the builder.
         */
        public Builder<TVector, TDistance> withM(int m) {
            this.m = m;
            return this;
        }

        /**
         * The parameter has the same meaning as ef, but controls the index time / index accuracy. Bigger efConstruction
         * leads to longer construction, but better index quality. At some point, increasing efConstruction does not
         * improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
         * a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
         * there is room for improvement.
         *
         * @param efConstruction controls the index time / index accuracy
         * @return the builder
         */
        public Builder<TVector, TDistance> withEfConstruction(int efConstruction) {
            this.efConstruction = efConstruction;
            return this;
        }

        /**
         * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
         * accurate but slower search. The value ef of can be anything between k and the size of the dataset.
         *
         * @param ef size of the dynamic list for the nearest neighbors
         * @return the builder
         */
        public Builder<TVector, TDistance> withEf(int ef) {
            this.ef = ef;
            return this;
        }

        /**
         * Call to enable support for the experimental remove operation. Indices that support removes will consume more
         * memory.
         *
         * @return the builder
         */
        public Builder<TVector, TDistance> withRemoveEnabled() {
            this.removeEnabled = true;
            return this;
        }

        /**
         * Build the index.
         *
         * @param <TId> type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the hnsw index instance
         */
        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<>(this);
        }

    }

}

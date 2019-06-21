package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.*;
import org.eclipse.collections.api.iterator.IntIterator;
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
 * Implementation of {@link Index} that implements the hnsw algorithm.
 *
 * @param <TId> Type of the external identifier of an item
 * @param <TVector> Type of the vector to perform distance calculation on
 * @param <TItem> Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 *
 * @see <a href="https://arxiv.org/abs/1603.09320">
 *     Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

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
    private final MutableIntStack freedIds;

    private volatile Node<TItem> entryPoint;

    private final AtomicReferenceArray<Node<TItem>> nodes;
    private final Map<TId, Integer> lookup;

    private final ObjectSerializer<TId> itemIdSerializer;
    private final ObjectSerializer<TItem> itemSerializer;

    private final ReentrantLock globalLock;

    private final Lock nonExclusiveLock;
    private final Lock exclusiveLock;

    private final Pool<VisitedBitSet> visitedBitSetPool;

    private HnswIndex(DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator,
                      int maxItemCount, int m, int maxM, int maxM0, double levelLambda, int ef, int efConstruction,
                      boolean removeEnabled, int itemCount, MutableIntStack freedIds, Node<TItem> entryPoint,
                      AtomicReferenceArray<Node<TItem>> nodes, Map<TId, Integer> lookup,
                      ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

        this.distanceFunction = distanceFunction;
        this.distanceComparator = distanceComparator;
        this.maxItemCount = maxItemCount;
        this.m = m;
        this.maxM = maxM;
        this.maxM0 = maxM0;
        this.levelLambda = levelLambda;
        this.ef = ef;
        this.efConstruction = efConstruction;
        this.removeEnabled = removeEnabled;
        this.itemCount = itemCount;
        this.freedIds = freedIds;
        this.entryPoint = entryPoint;
        this.nodes = nodes;
        this.lookup = lookup;
        this.itemIdSerializer = itemIdSerializer;
        this.itemSerializer = itemSerializer;

        this.globalLock = new ReentrantLock();

        ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        this.nonExclusiveLock = readWriteLock.readLock();
        this.exclusiveLock = readWriteLock.writeLock();

        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());
    }

    private HnswIndex(RefinedBuilder<TId, TVector, TItem, TDistance> builder) {

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

        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ConcurrentHashMap<>();

        this.freedIds = new IntArrayStack();

        this.itemIdSerializer = builder.itemIdSerializer;
        this.itemSerializer = builder.itemSerializer;

        this.globalLock = new ReentrantLock();

        ReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        this.nonExclusiveLock = readWriteLock.readLock();
        this.exclusiveLock = readWriteLock.writeLock();


        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount),
                Runtime.getRuntime().availableProcessors());
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
        exclusiveLock.lock();

        try {

            if (!removeEnabled) {
                return false;
            }

            Integer internalNodeId = lookup.get(id);

            if (internalNodeId == null) {
                return false;
            }

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

            lookup.remove(id);
            nodes.set(internalNodeId, null);

            synchronized (freedIds) {
                freedIds.push(internalNodeId);
            }
            return true;

        } finally {
            exclusiveLock.unlock();
        }


        // TODO do we want to do anything to fix up the connections like here https://github.com/andrusha97/online-hnsw/blob/master/include/hnsw/index.hpp#L185
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(TItem item) {

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

        globalLock.lock();

        try {

            if (lookup.containsKey(item.id())) {
                remove(item.id());
            }

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

            Node<TItem> newNode = new Node<>(newNodeId, outgoingConnections, incomingConnections, item);
            nodes.set(newNodeId, newNode);
            lookup.put(item.id(), newNodeId);

            nonExclusiveLock.lock();

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
                nonExclusiveLock.unlock();
            }
        } finally {
            if (globalLock.isHeldByCurrentThread()) {
                globalLock.unlock();
            }
        }
    }


    private void mutuallyConnectNewElement(Node<TItem> newNode,
                                           PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates,
                                           int level) {

        int bestN = level == 0 ? this.maxM0 : this.maxM;

        int newNodeId = newNode.id;
        TVector newItemVector = newNode.item.vector();
        MutableIntList outgoingNewItemConnections = newNode.outgoingConnections[level];

        getNeighborsByHeuristic2(topCandidates, null, m);

        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId;

            outgoingNewItemConnections.add(selectedNeighbourId);

            Node<TItem> neighbourNode = nodes.get(selectedNeighbourId);

            MutableIntList prunedConnections = null;

            if (removeEnabled) {
                prunedConnections = new IntArrayList();
            }

            synchronized (neighbourNode) {

                if (removeEnabled) {
                    neighbourNode.incomingConnections[level].add(newNodeId);
                }

                TVector neighbourVector = neighbourNode.item.vector();

                MutableIntList outgoingNeighbourConnectionsAtLevel = neighbourNode.outgoingConnections[level];

                if (outgoingNeighbourConnectionsAtLevel.size() < bestN) {

                    if (removeEnabled) {
                        newNode.incomingConnections[level].add(selectedNeighbourId);
                    }

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

                    getNeighborsByHeuristic2(candidates, prunedConnections, bestN);

                    if (removeEnabled) {
                        newNode.incomingConnections[level].add(selectedNeighbourId);
                    }

                    outgoingNeighbourConnectionsAtLevel.clear();

                    while(!candidates.isEmpty()) {
                        outgoingNeighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }

            if (removeEnabled) {
                prunedConnections.forEach(id -> {
                    Node<TItem> node = nodes.get(id);
                    synchronized (node) {
                        node.incomingConnections[level].remove(selectedNeighbourId);
                    }
                });
            }
        }
    }

    private void getNeighborsByHeuristic2(PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates,
                                          MutableIntList prunedConnections,
                                          int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<NodeIdAndDistance<TDistance>> queueClosest = new PriorityQueue<>();
        List<NodeIdAndDistance<TDistance>> returnList = new ArrayList<>();

        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while(!queueClosest.isEmpty()) {
            NodeIdAndDistance<TDistance> currentPair = queueClosest.poll();

            boolean good;
            if (returnList.size() >= m) {
                good = false;
            } else {
                TDistance distToQuery = currentPair.distance;

                good = true;
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
            }
            if (good) {
                returnList.add(currentPair);
            } else {
                if (prunedConnections != null) {
                    prunedConnections.add(currentPair.nodeId);
                }
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
     * such can be used as a baseline for assessing the precision of the index.
     * Searches will be really slow but give the correct result every time.
     *
     * @return read only view on top of this index that uses pairwise comparision when doing distance search
     */
    public Index<TId, TVector, TItem, TDistance> asExactIndex() {
        return new ExactIndex();
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
     * Returns the parameter has the same meaning as ef, but controls the index time / index precision.
     *
     * @return the parameter has the same meaning as ef, but controls the index time / index precision
     */
    public int getEfConstruction() {
        return efConstruction;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void save(OutputStream out) throws IOException {
        exclusiveLock.lock();

        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {

            oos.writeObject(distanceFunction);
            oos.writeObject(distanceComparator);
            oos.writeObject(itemIdSerializer);
            oos.writeObject(itemSerializer);

            oos.writeInt(maxItemCount);
            oos.writeInt(m);
            oos.writeInt(maxM);
            oos.writeInt(maxM0);
            oos.writeDouble(levelLambda);
            oos.writeInt(ef);
            oos.writeInt(efConstruction);
            oos.writeBoolean(removeEnabled);
            oos.writeInt(itemCount);

            writeMutableIntStack(oos, freedIds);

            writeNode(oos, entryPoint);
            writeNodes(oos, nodes);
            writeLookup(oos, lookup);
        } finally {
            exclusiveLock.unlock();
        }
    }

    private void writeLookup(ObjectOutputStream oos, Map<TId, Integer> lookup) throws IOException {
        Set<Map.Entry<TId, Integer>> entries = lookup.entrySet();

        oos.writeInt(entries.size());

        for (Map.Entry<TId, Integer> entry : entries) {
            itemIdSerializer.write(entry.getKey(), oos);
            oos.writeInt(entry.getValue());
        }
    }

    private void writeMutableIntStack(ObjectOutputStream oos, MutableIntStack stack) throws IOException {
        IntIterator iterator = stack.intIterator();

        oos.writeInt(stack.size());

        while(iterator.hasNext()) {
            oos.writeInt(iterator.next());
        }
    }

    private void writeNodes(ObjectOutputStream oos, AtomicReferenceArray<Node<TItem>> nodes) throws IOException {
        oos.writeInt(nodes.length());

        for (int i = 0; i < nodes.length(); i++) {
            writeNode(oos, nodes.get(i));
        }
    }

    private void writeNode(ObjectOutputStream oos, Node<TItem> node) throws IOException {
        if (node == null) {
            oos.writeInt(-1);
        } else {
            oos.writeInt(node.id);
            oos.writeInt(node.outgoingConnections.length);

            for (MutableIntList connections : node.outgoingConnections) {
                oos.writeInt(connections.size());
                for (int j = 0; j < connections.size(); j++) {
                    oos.writeInt(connections.get(j));
                }
            }

            itemSerializer.write(node.item, oos);

            if (removeEnabled) {
                oos.writeInt(node.incomingConnections.length);
                for (MutableIntList connections : node.incomingConnections) {
                    oos.writeInt(connections.size());
                    for (int j = 0; j < connections.size(); j++) {
                        oos.writeInt(connections.get(j));
                    }
                }
            }
        }
    }

    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file File to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            HnswIndex<TId, TVector, TItem, TDistance> load(File file) throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path Path to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            HnswIndex<TId, TVector, TItem, TDistance> load(Path path) throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream)
                throws IOException {

        try(ObjectInputStream ois = new ObjectInputStream(inputStream)) {

            DistanceFunction<TVector, TDistance> distanceFunction = (DistanceFunction<TVector, TDistance>) ois.readObject();
            Comparator<TDistance> distanceComparator = (Comparator<TDistance>) ois.readObject();
            ObjectSerializer<TId> itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
            ObjectSerializer<TItem> itemSerializer = (ObjectSerializer<TItem>) ois.readObject();

            int maxItemCount = ois.readInt();
            int m = ois.readInt();
            int maxM = ois.readInt();
            int maxM0 = ois.readInt();
            double levelLambda = ois.readDouble();
            int ef = ois.readInt();
            int efConstruction = ois.readInt();
            boolean removeEnabled = ois.readBoolean();
            int itemCount = ois.readInt();

            IntArrayStack freedIds = readIntArrayStack(ois);

            Node<TItem> entryPoint = readNode(ois, itemSerializer, maxM0, maxM, removeEnabled);

            AtomicReferenceArray<Node<TItem>> nodes = readNodes(ois, itemSerializer, maxM0, maxM, removeEnabled);

            Map<TId, Integer> lookup = readLookup(ois, itemIdSerializer);

            return new HnswIndex<>(distanceFunction, distanceComparator, maxItemCount, m, maxM, maxM0, levelLambda, ef,
                    efConstruction, removeEnabled, itemCount, freedIds, entryPoint, nodes, lookup, itemIdSerializer,
                    itemSerializer);

        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
    }

    private static IntArrayStack readIntArrayStack(ObjectInputStream ois) throws IOException {
        int size = ois.readInt();

        int[] values = new int[size];

        for (int i = 0; i < size; i++) {
            values[i] = ois.readInt();
        }
        return IntArrayStack.newStackWith(values);
    }

    private static IntArrayList readIntArrayList(ObjectInputStream ois, int initialSize) throws IOException {
        int size = ois.readInt();

        IntArrayList list = new IntArrayList(initialSize);

        for (int j = 0; j < size; j++) {
            list.add(ois.readInt());
        }

        return list;
    }

    private static <TItem> Node<TItem> readNode(ObjectInputStream ois,
                                                ObjectSerializer<TItem> itemSerializer,
                                                int maxM0,
                                                int maxM,
                                                boolean removeEnabled) throws IOException, ClassNotFoundException {

        int id = ois.readInt();

        if (id == -1) {
            return null;
        } else {
            int outgoingConnectionsSize = ois.readInt();

            MutableIntList[] outgoingConnections = new MutableIntList[outgoingConnectionsSize];

            for (int i = 0; i < outgoingConnectionsSize; i++) {
                int levelM = i == 0 ? maxM0 : maxM;
                outgoingConnections[i] = readIntArrayList(ois, levelM);
            }

            TItem item = itemSerializer.read(ois);

            MutableIntList[] incomingConnections = null;

            if (removeEnabled) {

                int incomingConnectionsSize = ois.readInt();
                incomingConnections = new MutableIntList[incomingConnectionsSize];

                for (int i = 0; i < incomingConnectionsSize; i++) {
                    int levelM = i == 0 ? maxM0 : maxM;
                    incomingConnections[i] = readIntArrayList(ois, levelM);
                }
            }

            return new Node<>(id, outgoingConnections, incomingConnections, item);
        }
    }

    private static final <TItem> AtomicReferenceArray<Node<TItem>> readNodes(ObjectInputStream ois,
                                                                             ObjectSerializer<TItem> itemSerializer,
                                                                             int maxM0,
                                                                             int maxM,
                                                                             boolean removeEnabled)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();
        AtomicReferenceArray<Node<TItem>> nodes = new AtomicReferenceArray<>(size);

        for (int i = 0; i < nodes.length(); i++) {
            nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM, removeEnabled));
        }

        return nodes;
    }

    private static final <TId> Map<TId, Integer> readLookup(ObjectInputStream ois,
                                                            ObjectSerializer<TId> itemIdSerializer)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();

        Map<TId, Integer> map = new ConcurrentHashMap<>(size);

        for (int i = 0; i < size; i++) {
            TId key = itemIdSerializer.read(ois);
            int value = ois.readInt();

            map.put(key, value);
        }
        return map;
    }

    public static <TVector, TDistance extends Comparable<TDistance>>
        Builder<TVector, TDistance>
            newBuilder(DistanceFunction<TVector, TDistance> distanceFunction, int maxItemCount) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

        return new Builder<>(distanceFunction, distanceComparator, maxItemCount);
    }

    public static <TVector, TDistance>
        Builder<TVector, TDistance>
            newBuilder(DistanceFunction<TVector, TDistance> distanceFunction, Comparator<TDistance> distanceComparator,
                       int maxItemCount) {

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

    class ExactIndex implements Index<TId, TVector, TItem, TDistance> {
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

        @Override
        public void add(TItem item) {
            HnswIndex.this.add(item);
        }

        @Override
        public boolean remove(TId id) {
            return HnswIndex.this.remove(id);
        }

        @Override
        public List<SearchResult<TItem, TDistance>> findNeighbors(TId id, int k) {
            return HnswIndex.this.findNeighbors(id, k);
        }

        @Override
        public void save(OutputStream out) throws IOException {
            HnswIndex.this.save(out);
        }

        @Override
        public void save(File file) throws IOException {
            HnswIndex.this.save(file);
        }

        @Override
        public void save(Path path) throws IOException {
            HnswIndex.this.save(path);
        }

        @Override
        public void addAll(Collection<TItem> items) throws InterruptedException {
            HnswIndex.this.addAll(items);
        }

        @Override
        public void addAll(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
            HnswIndex.this.addAll(items, listener);
        }

        @Override
        public void addAll(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval) throws InterruptedException {
            HnswIndex.this.addAll(items, numThreads, listener, progressUpdateInterval);
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

    public static abstract class BuilderBase<TBuilder extends BuilderBase<TBuilder, TVector, TDistance>, TVector, TDistance> {

        public static final int DEFAULT_M = 10;
        public static final int DEFAULT_EF = 10;
        public static final int DEFAULT_EF_CONSTRUCTION = 200;
        public static final boolean DEFAULT_REMOVE_ENABLED = false;

        DistanceFunction<TVector, TDistance> distanceFunction;
        Comparator<TDistance> distanceComparator;

        int maxItemCount;

        int m = DEFAULT_M;
        int ef = DEFAULT_EF;
        int efConstruction = DEFAULT_EF_CONSTRUCTION;
        boolean removeEnabled = DEFAULT_REMOVE_ENABLED;

        BuilderBase(DistanceFunction<TVector, TDistance> distanceFunction,
                    Comparator<TDistance> distanceComparator,
                    int maxItemCount) {

            this.distanceFunction = distanceFunction;
            this.distanceComparator = distanceComparator;
            this.maxItemCount = maxItemCount;
        }

        abstract TBuilder self();

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
        public TBuilder withM(int m) {
            this.m = m;
            return self();
        }

        /**`
         * The parameter has the same meaning as ef, but controls the index time / index precision. Bigger efConstruction
         * leads to longer construction, but better index quality. At some point, increasing efConstruction does not
         * improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure
         * a recall for M nearest neighbor search when ef = efConstruction: if the recall is lower than 0.9, then
         * there is room for improvement.
         *
         * @param efConstruction controls the index time / index precision
         * @return the builder
         */
        public TBuilder withEfConstruction(int efConstruction) {
            this.efConstruction = efConstruction;
            return self();
        }

        /**
         * The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more
         * accurate but slower search. The value ef of can be anything between k and the size of the dataset.
         *
         * @param ef size of the dynamic list for the nearest neighbors
         * @return the builder
         */
        public TBuilder withEf(int ef) {
            this.ef = ef;
            return self();
        }

        /**
         * Call to enable support for the experimental remove operation. Indices that support removes will consume more
         * memory.
         *
         * @return the builder
         */
        public TBuilder withRemoveEnabled() {
            this.removeEnabled = true;
            return self();
        }
    }


    /**
     * Builder for initializing an {@link HnswIndex} instance.
     *
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder<TVector, TDistance> extends BuilderBase<Builder<TVector, TDistance>, TVector, TDistance> {

        /**
         * Constructs a new {@link Builder} instance.
         *
         * @param distanceFunction the distance function
         * @param maxItemCount the maximum number of elements in the index
         */
        Builder(DistanceFunction<TVector, TDistance> distanceFunction,
                Comparator<TDistance> distanceComparator,
                int maxItemCount) {

            super(distanceFunction, distanceComparator, maxItemCount);
        }

        @Override
        Builder<TVector, TDistance> self() {
            return this;
        }

        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer serializes the
         * @param <TId> Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the builder
         */
        public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance>
                withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
            return new RefinedBuilder<>(distanceFunction, distanceComparator, maxItemCount, m, ef, efConstruction,
                    removeEnabled, itemIdSerializer, itemSerializer);
        }

        /**
         * Build the index.
         *
         * @param <TId> Type of the external identifier of an item
         * @param <TItem> implementation of the Item interface
         * @return the hnsw index instance
         */
        public <TId, TItem extends Item<TId, TVector>> HnswIndex<TId, TVector, TItem, TDistance> build() {
            ObjectSerializer<TId> itemIdSerializer = new JavaObjectSerializer<>();
            ObjectSerializer<TItem> itemSerializer = new JavaObjectSerializer<>();

            return withCustomSerializers(itemIdSerializer, itemSerializer)
                    .build();
        }

    }

    public static class RefinedBuilder<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
            extends BuilderBase<RefinedBuilder<TId, TVector, TItem, TDistance>, TVector, TDistance> {

        private ObjectSerializer<TId> itemIdSerializer;
        private ObjectSerializer<TItem> itemSerializer;

        RefinedBuilder(DistanceFunction<TVector, TDistance> distanceFunction,
                       Comparator<TDistance> distanceComparator,
                       int maxItemCount,
                       int m,
                       int ef,
                       int efConstruction,
                       boolean removeEnabled,
                       ObjectSerializer<TId> itemIdSerializer,
                       ObjectSerializer<TItem> itemSerializer) {

            super(distanceFunction, distanceComparator, maxItemCount);

            this.m = m;
            this.ef = ef;
            this.efConstruction = efConstruction;
            this.removeEnabled = removeEnabled;

            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;
        }

        @Override
        RefinedBuilder<TId, TVector, TItem, TDistance> self() {
            return this;
        }

        /**
         * Register the serializers used when saving the index.
         *
         * @param itemIdSerializer serializes the key of the item
         * @param itemSerializer serializes the
         * @return the builder
         */
        public RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(
                ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {

            this.itemIdSerializer = itemIdSerializer;
            this.itemSerializer = itemSerializer;

            return this;
        }

        /**
         * Build the index.
         *
         * @return the hnsw index instance
         */
        public HnswIndex<TId, TVector, TItem, TDistance> build() {
            return new HnswIndex<>(this);
        }

    }

}

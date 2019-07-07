package com.github.jelmerk.knn.hnsw;


import com.github.jelmerk.knn.*;
import com.github.jelmerk.knn.util.ArrayBitSet;
import com.github.jelmerk.knn.util.BitSet;
import com.github.jelmerk.knn.util.SynchronizedBitSet;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.*;

/**
 * Implementation of {@link Index} that implements the hnsw algorithm.
 *
 * @param <TId>       Type of the external identifier of an item
 * @param <TVector>   Type of the vector to perform distance calculation on
 * @param <TItem>     Type of items stored in the index
 * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
 * @see <a href="https://arxiv.org/abs/1603.09320">
 * Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs</a>
 */
public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance>
        implements Index<TId, TVector, TItem, TDistance> {

    private static final long serialVersionUID = 1L;

    private DistanceFunction<TVector, TDistance> distanceFunction;
    private Comparator<TDistance> distanceComparator;

    private int maxNodeCount;
    private int m;
    private int maxM;
    private int maxM0;
    private double levelLambda;
    private int ef;
    private int efConstruction;
    private boolean removeEnabled;

    private volatile int nodeCount;
    private volatile int removedNodeCount;

    private volatile Node<TItem> entryPoint;

    private AtomicReferenceArray<Node<TItem>> nodes;
    private Map<TId, Integer> lookup;

    private ObjectSerializer<TId> itemIdSerializer;
    private ObjectSerializer<TItem> itemSerializer;

    private ReentrantLock globalLock;

    private GenericObjectPool<BitSet> visitedBitSetPool;

    private BitSet activeConstruction;

    private HnswIndex(RefinedBuilder<TId, TVector, TItem, TDistance> builder) {

        this.maxNodeCount = builder.maxItemCount;
        this.distanceFunction = builder.distanceFunction;
        this.distanceComparator = new MaxValueComparator<>(builder.distanceComparator);

        this.m = builder.m;
        this.maxM = builder.m;
        this.maxM0 = builder.m * 2;
        this.levelLambda = 1 / Math.log(this.m);
        this.efConstruction = Math.max(builder.efConstruction, m);
        this.ef = builder.ef;
        this.removeEnabled = builder.removeEnabled;

        this.nodes = new AtomicReferenceArray<>(this.maxNodeCount);

        this.lookup = new ConcurrentHashMap<>();

        this.itemIdSerializer = builder.itemIdSerializer;
        this.itemSerializer = builder.itemSerializer;

        this.globalLock = new ReentrantLock();

        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxNodeCount),
                Runtime.getRuntime().availableProcessors());

        this.activeConstruction = new SynchronizedBitSet(new ArrayBitSet(this.maxNodeCount));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return nodeCount - removedNodeCount;
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
        globalLock.lock();

        try {
            Integer internalId = lookup.get(id);

            if (internalId == null) {
                return false;
            }

            Node<TItem> node = nodes.get(internalId);

            synchronized (node) {
                node.deleted = true;
                lookup.remove(id);
            }

            removedNodeCount++;

            return true;
        } finally {
            globalLock.unlock();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void add(TItem item) {

        int randomLevel = assignLevel(item.id(), this.levelLambda);

        IntArrayList[] connections = new IntArrayList[randomLevel + 1];

        for (int level = 0; level <= randomLevel; level++) {
            int levelM = randomLevel == 0 ? maxM0 : maxM;
            connections[level] = new IntArrayList(levelM);
        }

        globalLock.lock();

        try {

            Integer existingNodeId = lookup.get(item.id());

            if (existingNodeId != null) {
                if (removeEnabled) {
                    Node<TItem> node = nodes.get(existingNodeId);

                    synchronized (node) {
                        if (Objects.deepEquals(node.item.vector(), item.vector())) {
                            node.item = item;
                            return;
                        } else{
                            remove(item.id());
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Duplicate item : " + item.id());
                }
            }

            if (nodeCount >= this.maxNodeCount) {
                throw new IllegalStateException("The number of elements exceeds the specified limit.");
            }

            Node<TItem> entryPointCopy = entryPoint;

            int newNodeId = nodeCount++;

            Node<TItem> newNode = new Node<>(newNodeId, connections, item, false);

            nodes.set(newNodeId, newNode);
            lookup.put(item.id(), newNodeId);

            if (entryPoint != null && randomLevel <= entryPoint.maxLevel()) {
                globalLock.unlock();
            }

            activeConstruction.add(newNodeId);

            try {
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
                                        MutableIntList candidateConnections = currObj.connections[activeLevel];

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

                            if (entryPointCopy.deleted) {
                                TDistance distance = distanceFunction.distance(item.vector(), entryPointCopy.item.vector());
                                topCandidates.add(new NodeIdAndDistance<>(entryPointCopy.id, distance, distanceComparator));

                                if (topCandidates.size() > efConstruction) {
                                    topCandidates.poll();
                                }
                            }

                            mutuallyConnectNewElement(newNode, topCandidates, level);
                        }
                    }
                }

                // zoom out to the highest level
                if (entryPoint == null || newNode.maxLevel() > entryPointCopy.maxLevel()) {
                    // this is thread safe because we get the global lock when we add a level
                    this.entryPoint = newNode;
                }
            } finally {
                activeConstruction.remove(newNodeId);
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
        MutableIntList outgoingNewItemConnections = newNode.connections[level];

        getNeighborsByHeuristic2(topCandidates, m);

        while (!topCandidates.isEmpty()) {
            int selectedNeighbourId = topCandidates.poll().nodeId;

            if (activeConstruction.contains(selectedNeighbourId)) { // to avoid deadlocks
                continue;
            }

            outgoingNewItemConnections.add(selectedNeighbourId);

            Node<TItem> neighbourNode = nodes.get(selectedNeighbourId);

            synchronized (neighbourNode) {

                TVector neighbourVector = neighbourNode.item.vector();

                MutableIntList neighbourConnectionsAtLevel = neighbourNode.connections[level];

                if (neighbourConnectionsAtLevel.size() < bestN) {
                    neighbourConnectionsAtLevel.add(newNodeId);
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

                    neighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(
                                neighbourVector,
                                nodes.get(id).item.vector()
                        );

                        candidates.add(new NodeIdAndDistance<>(id, dist, distanceComparator));
                    });


                    getNeighborsByHeuristic2(candidates, bestN);

                    neighbourConnectionsAtLevel.clear();

                    while (!candidates.isEmpty()) {
                        neighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
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
    public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {

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
                    MutableIntList candidateConnections = currObj.connections[activeLevel];

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

        while (topCandidates.size() > k) {
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

        BitSet visitedBitSet = visitedBitSetPool.borrowObject();

        try {
            PriorityQueue<NodeIdAndDistance<TDistance>> topCandidates =
                    new PriorityQueue<>(Comparator.<NodeIdAndDistance<TDistance>>naturalOrder().reversed());
            PriorityQueue<NodeIdAndDistance<TDistance>> candidateSet = new PriorityQueue<>();

            TDistance lowerBound;

            if (!entryPointNode.deleted) {
                TDistance distance = distanceFunction.distance(destination, entryPointNode.item.vector());
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, distance, distanceComparator);

                topCandidates.add(pair);
                lowerBound = distance;
                candidateSet.add(pair);

            } else {
                lowerBound = MaxValueComparator.maxValue();
                NodeIdAndDistance<TDistance> pair = new NodeIdAndDistance<>(entryPointNode.id, lowerBound, distanceComparator);
                candidateSet.add(pair);
            }

            visitedBitSet.add(entryPointNode.id);

            while (!candidateSet.isEmpty()) {

                NodeIdAndDistance<TDistance> currentPair = candidateSet.poll();

                if (gt(currentPair.distance, lowerBound)) {
                    break;
                }

                Node<TItem> node = nodes.get(currentPair.nodeId);

                synchronized (node) {

                    MutableIntList candidates = node.connections[layer];

                    for (int i = 0; i < candidates.size(); i++) {

                        int candidateId = candidates.get(i);

                        if (!visitedBitSet.contains(candidateId)) {

                            visitedBitSet.add(candidateId);

                            Node<TItem> candidateNode = nodes.get(candidateId);

                            TDistance candidateDistance = distanceFunction.distance(destination,
                                    candidateNode.item.vector());

                            if (topCandidates.size() < k || gt(lowerBound, candidateDistance)) {

                                NodeIdAndDistance<TDistance> candidatePair =
                                        new NodeIdAndDistance<>(candidateId, candidateDistance, distanceComparator);

                                candidateSet.add(candidatePair);

                                if (!candidateNode.deleted) {
                                    topCandidates.add(candidatePair);
                                }

                                if (topCandidates.size() > k) {
                                    topCandidates.poll();
                                }

                                if (!topCandidates.isEmpty()) {
                                    lowerBound = topCandidates.peek().distance;
                                }
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

        // TODO need to make sure no adds are happening when saving .. how use the stampedlock anyway ?

        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(this);
        }
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        oos.writeObject(distanceFunction);
        oos.writeObject(distanceComparator);
        oos.writeObject(itemIdSerializer);
        oos.writeObject(itemSerializer);

        oos.writeInt(maxNodeCount);
        oos.writeInt(m);
        oos.writeInt(maxM);
        oos.writeInt(maxM0);
        oos.writeDouble(levelLambda);
        oos.writeInt(ef);
        oos.writeInt(efConstruction);
        oos.writeBoolean(removeEnabled);
        oos.writeInt(nodeCount);
        oos.writeInt(removedNodeCount);

        writeNode(oos, entryPoint);
        writeNodes(oos, nodes);
        writeLookup(oos, lookup);
    }

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException {
        this.distanceFunction = (DistanceFunction<TVector, TDistance>) ois.readObject();
        this.distanceComparator = (Comparator<TDistance>) ois.readObject();
        this.itemIdSerializer = (ObjectSerializer<TId>) ois.readObject();
        this.itemSerializer = (ObjectSerializer<TItem>) ois.readObject();

        this.maxNodeCount = ois.readInt();
        this.m = ois.readInt();
        this.maxM = ois.readInt();
        this.maxM0 = ois.readInt();
        this.levelLambda = ois.readDouble();
        this.ef = ois.readInt();
        this.efConstruction = ois.readInt();
        this.removeEnabled = ois.readBoolean();
        this.nodeCount = ois.readInt();
        this.removedNodeCount = ois.readInt();

        this.entryPoint = readNode(ois, itemSerializer, maxM0, maxM);

        this.nodes = readNodes(ois, itemSerializer, maxM0, maxM);

        this.lookup = readLookup(ois, itemIdSerializer);

        this.globalLock = new ReentrantLock();

        this.visitedBitSetPool = new GenericObjectPool<>(() -> new ArrayBitSet(this.maxNodeCount),
                Runtime.getRuntime().availableProcessors());

        this.activeConstruction = new SynchronizedBitSet(new ArrayBitSet(this.maxNodeCount));

    }

    private void writeLookup(ObjectOutputStream oos, Map<TId, Integer> lookup) throws IOException {
        Set<Map.Entry<TId, Integer>> entries = lookup.entrySet();

        oos.writeInt(entries.size());

        for (Map.Entry<TId, Integer> entry : entries) {
            itemIdSerializer.write(entry.getKey(), oos);
            oos.writeInt(entry.getValue());
        }
    }

    private void writeNodes(ObjectOutputStream oos, AtomicReferenceArray<Node<TItem>> nodes) throws IOException {
        oos.writeInt(nodes.length());

        for (int i = 0; i < nodes.length(); i++) {
            writeNode(oos, nodes.get(i));
        }
    }

    private void writeNode(ObjectOutputStream oos, Node<TItem> node) throws IOException {

        // TODO can i do this nicer.. like not iterate over every node in the array but only the ones with non null values
        if (node == null) {
            oos.writeInt(-1);
        } else {
            oos.writeInt(node.id);
            oos.writeBoolean(node.deleted);
            oos.writeInt(node.connections.length);

            for (MutableIntList connections : node.connections) {
                oos.writeInt(connections.size());
                for (int j = 0; j < connections.size(); j++) {
                    oos.writeInt(connections.get(j));
                }
            }

            itemSerializer.write(node.item, oos);

        }
    }

    /**
     * Restores a {@link HnswIndex} from a File.
     *
     * @param file        File to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(File file)
            throws IOException {
        return load(new FileInputStream(file));
    }

    /**
     * Restores a {@link HnswIndex} from a Path.
     *
     * @param path        Path to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return The restored index
     * @throws IOException in case of an I/O exception
     */
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(Path path)
            throws IOException {
        return load(Files.newInputStream(path));
    }

    /**
     * Restores a {@link HnswIndex} from an InputStream.
     *
     * @param inputStream InputStream to restore the index from
     * @param <TId>       Type of the external identifier of an item
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TItem>     Type of items stored in the index
     * @param <TDistance> The type of distance between items (expect any numeric type: float, double, int, ...).
     * @return The restored index
     * @throws IOException              in case of an I/O exception
     * @throws IllegalArgumentException in case the file cannot be read
     */
    @SuppressWarnings("unchecked")
    public static <TId, TVector, TItem extends Item<TId, TVector>, TDistance> HnswIndex<TId, TVector, TItem, TDistance> load(InputStream inputStream)
            throws IOException {

        try (ObjectInputStream ois = new ObjectInputStream(inputStream)) {
            return (HnswIndex<TId, TVector, TItem, TDistance>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IllegalArgumentException("Could not read input file.", e);
        }
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
                                                int maxM) throws IOException, ClassNotFoundException {

        int id = ois.readInt();
        if (id == -1) {
            return null;
        } else {
            boolean deleted = ois.readBoolean();
            int outgoingConnectionsSize = ois.readInt();

            MutableIntList[] connections = new MutableIntList[outgoingConnectionsSize];

            for (int i = 0; i < outgoingConnectionsSize; i++) {
                int levelM = i == 0 ? maxM0 : maxM;
                connections[i] = readIntArrayList(ois, levelM);
            }

            TItem item = itemSerializer.read(ois);

            return new Node<>(id, connections, item, deleted);
        }
    }

    private static <TItem> AtomicReferenceArray<Node<TItem>> readNodes(ObjectInputStream ois,
                                                                       ObjectSerializer<TItem> itemSerializer,
                                                                       int maxM0,
                                                                       int maxM)
            throws IOException, ClassNotFoundException {

        int size = ois.readInt();
        AtomicReferenceArray<Node<TItem>> nodes = new AtomicReferenceArray<>(size);

        for (int i = 0; i < nodes.length(); i++) {
            nodes.set(i, readNode(ois, itemSerializer, maxM0, maxM));
        }

        return nodes;
    }

    private static <TId> Map<TId, Integer> readLookup(ObjectInputStream ois,
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

    /**
     * Start the process of building a new HNSW index.
     *
     * @param distanceFunction the distance function
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance extends Comparable<TDistance>> Builder<TVector, TDistance> newBuilder(
            DistanceFunction<TVector, TDistance> distanceFunction,
            int maxItemCount) {

        Comparator<TDistance> distanceComparator = Comparator.naturalOrder();

        return new Builder<>(distanceFunction, distanceComparator, maxItemCount);
    }

    /**
     * Start the process of building a new HNSW index.
     *
     * @param distanceFunction the distance function
     * @param distanceComparator used to compare distances
     * @param maxItemCount maximum number of items the index can hold
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     * @return a builder
     */
    public static <TVector, TDistance> Builder<TVector, TDistance> newBuilder(
            DistanceFunction<TVector, TDistance> distanceFunction,
            Comparator<TDistance> distanceComparator,
            int maxItemCount) {

        return new Builder<>(distanceFunction, distanceComparator, maxItemCount);
    }

    private int assignLevel(TId value, double lambda) {

        // by relying on the external id to come up with the level, the graph construction should be a lot mor stable
        // see : https://github.com/nmslib/hnswlib/issues/28

        int hashCode = value.hashCode();

        byte[] bytes = new byte[]{
                (byte) (hashCode >> 24),
                (byte) (hashCode >> 16),
                (byte) (hashCode >> 8),
                (byte) hashCode
        };

        double random = Math.abs((double) Murmur3.hash32(bytes) / (double) Integer.MAX_VALUE);

        double r = -Math.log(random) * lambda;
        return (int) r;
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

            for (int i = 0; i < nodeCount; i++) {
                Node<TItem> node = nodes.get(i);
                if (node == null || node.deleted) {
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
            while ((result = queue.poll()) != null) { // if you iterate over a priority queue the order is not guaranteed
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

        final MutableIntList[] connections;

        volatile TItem item;

        volatile boolean deleted;

        Node(int id, MutableIntList[] connections, TItem item, boolean deleted) {
            this.id = id;
            this.connections = connections;
            this.item = item;
            this.deleted = deleted;
        }

        int maxLevel() {
            return this.connections.length - 1;
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
            return distanceComparator.compare(distance, o.distance);
        }
    }

    /**
     * Base class for HNSW index builders.
     *
     * @param <TBuilder> Concrete class that extends from this builder
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TDistance> Type of items stored in the index
     */
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

        /**
         * `
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
     * @param <TVector>   Type of the vector to perform distance calculation on
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
    public static class Builder<TVector, TDistance> extends BuilderBase<Builder<TVector, TDistance>, TVector, TDistance> {

        /**
         * Constructs a new {@link Builder} instance.
         *
         * @param distanceFunction the distance function
         * @param maxItemCount     the maximum number of elements in the index
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
         * @param itemSerializer   serializes the
         * @param <TId>            Type of the external identifier of an item
         * @param <TItem>          implementation of the Item interface
         * @return the builder
         */
        public <TId, TItem extends Item<TId, TVector>> RefinedBuilder<TId, TVector, TItem, TDistance> withCustomSerializers(ObjectSerializer<TId> itemIdSerializer, ObjectSerializer<TItem> itemSerializer) {
            return new RefinedBuilder<>(distanceFunction, distanceComparator, maxItemCount, m, ef, efConstruction,
                    removeEnabled, itemIdSerializer, itemSerializer);
        }

        /**
         * Build the index that uses java object serializers to store the items when reading and writing the index.
         *
         * @param <TId>   Type of the external identifier of an item
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

    /**
     * Extension of {@link Builder} that has knows what type of item is going to be stored in the index.
     *
     * @param <TId> Type of the external identifier of an item
     * @param <TVector> Type of the vector to perform distance calculation on
     * @param <TItem> Type of items stored in the index
     * @param <TDistance> Type of distance between items (expect any numeric type: float, double, int, ..)
     */
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
         * @param itemSerializer   serializes the
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

    static class MaxValueComparator<TDistance> implements Comparator<TDistance>, Serializable  {

        private final Comparator<TDistance> delegate;

        MaxValueComparator(Comparator<TDistance> delegate) {
            this.delegate = delegate;
        }

        @Override
        public int compare(TDistance o1, TDistance o2) {
            return o1 == null ? o2 == null ? 0 : 1
                    : o2 == null ? -1 : delegate.compare(o1, o2);
        }

        static <TDistance> TDistance maxValue() {
            return null;
        }
    }

}

package org.github.jelmerk.hnsw;


import org.eclipse.collections.api.list.MutableList;
import org.eclipse.collections.api.list.primitive.IntList;
import org.eclipse.collections.api.list.primitive.MutableIntList;
import org.eclipse.collections.api.set.primitive.MutableIntSet;
import org.eclipse.collections.impl.list.mutable.primitive.IntArrayList;
import org.eclipse.collections.impl.set.mutable.primitive.IntHashSet;
import org.github.jelmerk.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class HnswIndex<TId, TVector, TItem extends Item<TId, TVector>, TDistance extends Comparable<TDistance>>
        implements Index<TId, TVector, TItem, TDistance>, Serializable {

    private static final long serialVersionUID = 2779232910534910891L;

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

    private final Map<TId, Integer> lookup;

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
        this.constructionPruning = Math.max(builder.constructionPruning, m);
        this.expandBestSelection = builder.expandBestSelection;
        this.keepPrunedConnections =  builder.keepPrunedConnections;

        if (builder.neighbourHeuristic == NeighbourSelectionHeuristic.SELECT_SIMPLE) {
            this.algorithm = new Algorithm3();
        } else {
            this.algorithm = new Algorithm4();
        }

        this.random = new DotNetRandom(builder.randomSeed); // TODO JK: get rid of this dot net random and use a ThreadLocalRandom so we don't have to synchronize access

        this.globalLock = new ReentrantLock();

        this.itemCount = new AtomicInteger();
        this.items = new AtomicReferenceArray<>(this.maxItemCount);
        this.nodes = new AtomicReferenceArray<>(this.maxItemCount);

        this.lookup = new ConcurrentHashMap<>();

        this.visitedBitSetPool = new Pool<>(() -> new VisitedBitSet(this.maxItemCount), 12);
        this.expansionBufferPool = new Pool<>(IntArrayList::new, 12); // TODO i think we can get rid of this
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

        int count = itemCount.getAndIncrement(); // TODO JK i guess we don't want this count to increase if there's no space, how ?

        if (count >= this.maxItemCount) {
            throw new IllegalStateException("The number of elements exceeds the specified limit.");
        }

        items.set(count, item);

        Node newNode = this.algorithm.newNode(count, randomLayer(random, this.levelLambda));
        nodes.set(count, newNode);

        lookup.put(item.getId(), count);

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





                Node currObj = entrypointCopy;

                if (currObj != null) {

                    if (newNode.maxLayer() < entrypointCopy.maxLayer()) {

                        TDistance curDist = distanceFunction.distance(item.getVector(), items.get(currObj.id).getVector());

                        for (int activeLevel = entrypointCopy.maxLayer(); activeLevel > newNode.maxLayer(); activeLevel--) {

                            boolean changed = true;

                            while (changed) {
                                changed = false;

                                synchronized (currObj) {

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
                }

                int bestPeerId = currObj.id;


                // zoom in and find the best peer on the same level as newNode
                TravelingCosts<Integer, TDistance> currentNodeTravelingCosts = new TravelingCosts<>(this::calculateDistance, newNode.id);

//                int bestPeerId = findBestPeer(entrypointCopy.id, entrypointCopy.maxLayer(), newNode.maxLayer(), currentNodeTravelingCosts);

                MutableIntList neighboursIdsBuffer = new IntArrayList(algorithm.getM(0) + 1);

                // connecting new nodeId to the small world
                for (int layer = Math.min(newNode.maxLayer(), entrypointCopy.maxLayer()); layer >= 0; layer--) {
                    runKnnAtLayer(bestPeerId, currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.constructionPruning);
                    MutableIntList bestNeighboursIds = algorithm.selectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

                    for (int i = 0; i < bestNeighboursIds.size(); i++) {
                        int newNeighbourId = bestNeighboursIds.get(i);

                        Node neighbourNode;
                        synchronized (neighbourNode = nodes.get(newNeighbourId)) {
                            algorithm.connect(newNode, neighbourNode, layer); // JK this can mutate the new nodeId
                            algorithm.connect(neighbourNode, newNode, layer); // JK this can mutat the neighbour nodeId
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




    void addAll2(Collection<TItem> items) throws InterruptedException {
        addAll2(items, NullProgressListener.INSTANCE);
    }

    void addAll2(Collection<TItem> items, ProgressListener listener) throws InterruptedException {
        addAll2(items, Runtime.getRuntime().availableProcessors(), listener, DEFAULT_PROGRESS_UPDATE_INTERVAL);
    }

    void addAll2(Collection<TItem> items, int numThreads, ProgressListener listener, int progressUpdateInterval)
            throws InterruptedException {

        AtomicReference<RuntimeException> throwableHolder = new AtomicReference<>();

        ExecutorService executorService = Executors.newFixedThreadPool(numThreads);

        AtomicInteger workDone = new AtomicInteger();

        try {
            Queue<TItem> queue = new LinkedBlockingDeque<>(items);

            CountDownLatch latch = new CountDownLatch(numThreads);

            for (int threadId = 0; threadId < numThreads; threadId++) {

                executorService.submit(() -> {
                    TItem item;
                    while((item = queue.poll()) != null) {
                        try {
                            add2(item);

                            int done = workDone.incrementAndGet();

                            if (done % progressUpdateInterval == 0) {
                                listener.updateProgress(done, items.size());
                            }

                        } catch (RuntimeException t) {
                            throwableHolder.set(t);
                        }
                    }

                    latch.countDown();
                });
            }

            latch.await();

            RuntimeException throwable = throwableHolder.get();

            if (throwable != null) {
                throw throwable;
            }

        } finally {
            executorService.shutdown();
        }
    }

    public void add2(TItem item) {

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

//                        // TODO does using construction pruning make sense here why not just pass in bestN
                        PriorityQueue<DistanceNodePair<TDistance>> topCandidates =
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
                                           PriorityQueue<DistanceNodePair<TDistance>> topCandidates,
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

                    Comparator<DistanceNodePair<TDistance>> comparator = Comparator.<DistanceNodePair<TDistance>>naturalOrder().reversed();

                    PriorityQueue<DistanceNodePair<TDistance>> candidates = new PriorityQueue<>(comparator);
                    candidates.add(new DistanceNodePair<>(dMax, nodeId));

                    neighbourConnectionsAtLevel.forEach(id -> {
                        TDistance dist = distanceFunction.distance(items.get(selectedNeighbourId).getVector(), items.get(id).getVector());
                        candidates.add(new DistanceNodePair<>(dist, id));
                    });

                    getNeighborsByHeuristic2(candidates, bestN);
//                    getNeighborsByHeuristic1(candidates, bestN); // TODO jk, back to 2

                    // TODO more efficient than allocating a new array i guess but need to verify this
                    neighbourConnectionsAtLevel.clear();

                    while(!candidates.isEmpty()) {
                        neighbourConnectionsAtLevel.add(candidates.poll().nodeId);
                    }
                }
            }
        }
    }

    // TODO JK: not in the original hnsw impl but i think this is what it should be if you look at the algorithm3 class from the .net impl
    private void getNeighborsByHeuristic1(PriorityQueue<DistanceNodePair<TDistance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<DistanceNodePair<TDistance>> queueClosest = new PriorityQueue<>();

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

    private void getNeighborsByHeuristic2(PriorityQueue<DistanceNodePair<TDistance>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<DistanceNodePair<TDistance>> queueClosest = new PriorityQueue<>();
        List<DistanceNodePair<TDistance>> returnList = new ArrayList<>();

        while(!topCandidates.isEmpty()) {
            queueClosest.add(topCandidates.poll());
        }

        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            DistanceNodePair<TDistance> currentPair = queueClosest.poll();

            TDistance distToQuery = currentPair.distance;

            boolean good = true;
            for (DistanceNodePair<TDistance> secondPair : returnList) {

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

    public List<SearchResult<TItem, TDistance>>findNearest2(TVector destination, int k) {

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

        PriorityQueue<DistanceNodePair<TDistance>> topCandidates = searchBaseLayer(
                currObj.id, destination, k, 0);


        // TODO JK this code makes no sense if we have no ef value because the priority queue will never be bigger than k, delete it ? or work out what ef is for

        while(topCandidates.size() > k) {
            topCandidates.poll();
        }

        List<SearchResult<TItem, TDistance>> results = new ArrayList<>(topCandidates.size());
        while (!topCandidates.isEmpty()) {
            DistanceNodePair<TDistance> pair = topCandidates.poll();
            results.add(0, new SearchResult<>(items.get(pair.nodeId), pair.distance));
        }

        return results;
    }

    private PriorityQueue<DistanceNodePair<TDistance>> searchBaseLayer(
            int entryPointId, TVector destination, int k, int layer) {

        VisitedBitSet visitedBitSet = visitedBitSetPool.borrowObject();

        PriorityQueue<DistanceNodePair<TDistance>> topCandidates =
                new PriorityQueue<>(Comparator.<DistanceNodePair<TDistance>>naturalOrder().reversed());
        PriorityQueue<DistanceNodePair<TDistance>> candidateSet = new PriorityQueue<>();

        TDistance distance = distanceFunction.distance(destination, items.get(entryPointId).getVector());

        DistanceNodePair<TDistance> pair = new DistanceNodePair<>(distance, entryPointId);

        topCandidates.add(pair);
        candidateSet.add(pair);
        visitedBitSet.add(entryPointId);

        TDistance lowerBound = distance;

        while(!candidateSet.isEmpty()) {

            DistanceNodePair<TDistance> currentPair = candidateSet.peek();

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

                            DistanceNodePair<TDistance> candidatePair = new DistanceNodePair<>(candidateDistance, candidateId);

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
    public List<SearchResult<TItem, TDistance>> findNearest(TVector destination, int k) {

        TravelingCosts<Integer, TDistance> destinationTravelingCosts = new TravelingCosts<>(
                (x, y) -> this.distanceFunction.distance(destination, items.get(x).getVector())
        , -1);

        Node entrypointCopy = entryPoint;

        int bestPeerId = findBestPeer(entrypointCopy.id, entrypointCopy.maxLayer(), 0, destinationTravelingCosts);


//        Node currObj = entrypointCopy;
//
//        TDistance curDist = distanceFunction.distance(destination, items.get(currObj.id).getVector());
//
//        for (int activeLevel = entrypointCopy.maxLayer(); activeLevel > 0; activeLevel--) {
//
//            boolean changed = true;
//
//            while (changed) {
//                changed = false;
//
//                synchronized (currObj) {
//
//                    MutableIntList candidateConnections = currObj.connections[activeLevel - 1];
//
//                    for (int i = 0; i < candidateConnections.size(); i++) {
//
//                        int candidateId = candidateConnections.get(i);
//
//                        TDistance candidateDistance = distanceFunction.distance(destination, items.get(candidateId).getVector());
//                        if (DistanceUtils.lt(candidateDistance, curDist)) {
//                            curDist = candidateDistance;
//                            currObj = nodes.get(candidateId);
//                            changed = true;
//                        }
//                    }
//                }
//
//            }
//        }
//
//       int bestPeerId = currObj.id;

        MutableIntList resultIds = new IntArrayList(k + 1);
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
    public TItem remove(TId tId) {

        // TODO problem seems to be


        // what if it is the entry point

        Integer index = lookup.get(tId); // TODO make this a ObjectIntHashMap ?

        Node node = nodes.get(index);

        synchronized (node) {

            for (int layer = node.maxLayer(); layer >= 0; layer--) {


                final int finalLayer = layer;

                MutableIntList connections = node.connections[finalLayer];

                connections.forEach(connectedNodeId -> {
                    Node connectedNode = nodes.get(connectedNodeId);

                    synchronized (connectedNode) {

                        // remove the nodeId from the connections
                        // find the nodeId that should take its place

                        connectedNode.connections[finalLayer].remove(node.id);


                    }

                });

            }

            return items.get(index);
        }

    }




    /**
     * Prints edges of the graph.
     *
     * @return String representation of the graph's edges,
     */
    String print() {
        StringBuilder buffer = new StringBuilder();
        for (int layer = this.entryPoint.maxLayer(); layer >= 0; --layer) {
            buffer.append(String.format("[LEVEL %s]%n", layer));
            int finalLevel = layer;

            bfs(this.entryPoint, layer, node -> {

                String neighbours = node.connections[finalLevel].collect(String::valueOf).stream().map(String::valueOf)
                        .collect(Collectors.joining(","));
                buffer.append(String.format("(%d) -> {%s}%n", node.id, neighbours));

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
    private void bfs(Node entryPoint, int layer, Consumer<Node> visitConsumer) {

        MutableIntSet visitedIds = new IntHashSet();
        MutableIntList expansionQueue = IntArrayList.newListWith(entryPoint.id);

        while (!expansionQueue.isEmpty()) {
            Node currentNode = nodes.get(expansionQueue.removeAtIndex(0));
            if (!visitedIds.contains(currentNode.id)) {
                visitConsumer.accept(currentNode);
                visitedIds.add(currentNode.id);
                expansionQueue.addAll(currentNode.connections[layer]);
            }
        }
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

//                if (k == 20) {
//                    System.out.println(expansionHeap.getBuffer());
//                }
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
     * The implementation of the nodeId in hnsw graph.
     */
    static class Node implements Serializable {

        private int id;

        private MutableIntList[] connections;

        /**
         * Gets the max layer where the nodeId is presented.
         */
        int maxLayer() {
            return this.connections.length - 1;
        }
    }

    /**
     * The abstract class representing algorithm to control nodeId capacity.
     */
    abstract class Algorithm implements Serializable {

        /**
         * Creates a new instance of the {@link Node} struct. Controls the exact type of connection lists.
         *
         * @param nodeId The identifier of the nodeId.
         * @param maxLayer The max layer where the nodeId is presented.
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
         * The algorithm which selects best neighbours from the candidates for the given nodeId.
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
         * Tries to connect the nodeId with the new neighbour.
         *
         * @param node The nodeId to add neighbour to.
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


    static class DistanceNodePair<TDistance extends Comparable<TDistance>> implements Comparable<DistanceNodePair<TDistance>> {

        final TDistance distance;
        final int nodeId;

        public DistanceNodePair(TDistance distance, int nodeId) {
            this.distance = distance;
            this.nodeId = nodeId;
        }

        @Override
        public int compareTo(DistanceNodePair<TDistance> o) {
            return distance.compareTo(o.distance);
        }

        @Override
        public String toString() {

            return String.valueOf(nodeId);

//            return "DistanceNodePair{" +
//                    "distance=" + distance +
//                    ", nodeId=" + nodeId +
//                    '}';
        }
    }

}

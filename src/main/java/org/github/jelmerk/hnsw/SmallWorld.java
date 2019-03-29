package org.github.jelmerk.hnsw;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class SmallWorld<TItem, TDistance extends Comparable<TDistance>> {

    private BiFunction<TItem, TItem, TDistance> distance;

    /// <summary>
    /// The hierarchical small world graph instance.
    /// </summary>
    private Graph graph;

    public SmallWorld(BiFunction<TItem, TItem, TDistance> distance) {
        this.distance = distance;
    }


    public void buildGraph(List<TItem> items, DotNetRandom generator, Parameters parameters) {
        Graph graph = new Graph(this.distance, parameters);
        graph.create(items, generator);
        this.graph = graph;
    }


    /// <summary>
    /// Run knn search for a given item.
    /// </summary>
    /// <param name="item">The item to search nearest neighbours.</param>
    /// <param name="k">The number of nearest neighbours.</param>
    /// <returns>The list of found nearest neighbours.</returns>
    public List<KNNSearchResult> kNNSearch(TItem item, int k) {
        Node destination = this.graph.newNode.apply(-1, item, 0);
        List<Node> neighbourhood = this.graph.kNearest(destination, k);

        return neighbourhood.stream()
                .map(n -> {
                    KNNSearchResult result = new KNNSearchResult();
                    result.setId(n.getId());
                    result.setItem(n.getItem());
                    result.setDistance(destination.getTravelingCosts().from(n));
                    return result;
                })
                .collect(Collectors.toList());
    }


    public byte[] serializeGraph() {

        /*
            if (this.graph == null)
            {
                throw new InvalidOperationException("The graph does not exist");
            }

            var formatter = new BinaryFormatter();
            using (var stream = new MemoryStream())
            {
                formatter.Serialize(stream, this.graph.Parameters);

                var edgeBytes = this.graph.Serialize();
                stream.Write(edgeBytes, 0, edgeBytes.Length);

                return stream.ToArray();
            }
         */
        throw new UnsupportedOperationException("implement");
    }

    public void deserializeGraph(TItem[] items, byte[] bytes) {

        /*
            var formatter = new BinaryFormatter();
            using (var stream = new MemoryStream(bytes))
            {
                var parameters = (Parameters)formatter.Deserialize(stream);

                var graph = new Graph(this.distance, parameters);
                graph.Deserialize(items, bytes.Skip((int)stream.Position).ToArray());

                this.graph = graph;
            }

         */
        throw new UnsupportedOperationException("implement");
    }

    String print() {
        return this.graph.print();
    }


    public boolean dLt(TDistance x, TDistance y) {
        return x.compareTo(y) < 0;
    }

    public boolean dGt(TDistance x, TDistance y) {
        return x.compareTo(y) > 0;
    }

    public boolean dEq(TDistance x, TDistance y) {
        return x.compareTo(y) == 0;
    }

    public void bfs(Node entryPoint, int level, Consumer<Node> visitConsumer) {
        Set<Integer> visitedIds = new HashSet<>();

        Queue<Node> expansionQueue = new LinkedList<>(Collections.singleton(entryPoint));

        while (!expansionQueue.isEmpty()) {
            Node currentNode = expansionQueue.remove();
            if (!visitedIds.contains(currentNode.getId())) {
                visitConsumer.accept(currentNode);
                visitedIds.add(currentNode.getId());
                expansionQueue.addAll(currentNode.getConnections(level));
            }
        }
    }


    enum NeighbourSelectionHeuristic {
        SELECT_SIMPLE, SELECTION_HEURISTIC;
    }

    public static class Parameters {

        private int m = 10;
        private double levelLambda = 1 / Math.log(m);
        private NeighbourSelectionHeuristic neighbourHeuristic = NeighbourSelectionHeuristic.SELECT_SIMPLE;
        private int constructionPruning = 200;
        private boolean expandBestSelection = false;
        private boolean keepPrunedConnections = true;

        public int getM() {
            return m;
        }

        public void setM(int m) {
            this.m = m;
        }

        public double getLevelLambda() {
            return levelLambda;
        }

        public void setLevelLambda(double levelLambda) {
            this.levelLambda = levelLambda;
        }

        public NeighbourSelectionHeuristic getNeighbourHeuristic() {
            return neighbourHeuristic;
        }

        public void setNeighbourHeuristic(NeighbourSelectionHeuristic neighbourHeuristic) {
            this.neighbourHeuristic = neighbourHeuristic;
        }

        public int getConstructionPruning() {
            return constructionPruning;
        }

        public void setConstructionPruning(int constructionPruning) {
            this.constructionPruning = constructionPruning;
        }

        public boolean isExpandBestSelection() {
            return expandBestSelection;
        }

        public void setExpandBestSelection(boolean expandBestSelection) {
            this.expandBestSelection = expandBestSelection;
        }

        public boolean isKeepPrunedConnections() {
            return keepPrunedConnections;
        }

        public void setKeepPrunedConnections(boolean keepPrunedConnections) {
            this.keepPrunedConnections = keepPrunedConnections;
        }
    }


    class KNNSearchResult {

        private int id;
        private TItem item;
        private TDistance Distance;

        public int getId() {
            return id;
        }

        public void setId(int id) {
            this.id = id;
        }

        public TItem getItem() {
            return item;
        }

        public void setItem(TItem item) {
            this.item = item;
        }

        public TDistance getDistance() {
            return Distance;
        }

        public void setDistance(TDistance distance) {
            Distance = distance;
        }

        @Override
        public String toString() {
            return "KNNSearchResult{" +
                    "id=" + id +
                    ", item=" + item +
                    ", Distance=" + Distance +
                    '}';
        }
    }


    abstract class Node {

        private int id;
        private TItem item;
        private int maxLevel;
        private BiFunction<TItem, TItem, TDistance> distance;
        private Parameters parameters;

        private List<List<Node>> connections;
        private TravelingCosts<Node, TDistance> travelingCosts;


        Node(int id, TItem item, int maxLevel, BiFunction<TItem, TItem, TDistance> distance, Parameters parameters) {
            this.id = id;
            this.item = item;
            this.maxLevel = maxLevel;
            this.distance = distance;
            this.parameters = parameters;

            this.connections = new ArrayList<>(this.maxLevel + 1);
            for (int level = 0; level <= maxLevel; level++) {
                this.connections.add(new ArrayList<>(getM(this.parameters.m, level)));
            }

            BiFunction<Node, Node, TDistance> nodesDistance = (Node x, Node y) -> distance.apply(x.getItem(), y.getItem());
            this.travelingCosts = new TravelingCosts<>(nodesDistance, this);
        }

        public int getId() {
            return id;
        }

        public TItem getItem() {
            return item;
        }

        public int getMaxLevel() {
            return maxLevel;
        }

        public BiFunction<TItem, TItem, TDistance> getDistance() {
            return distance;
        }

        public TravelingCosts<Node, TDistance> getTravelingCosts() {
            return travelingCosts;
        }

        protected Parameters getParameters() {
            return parameters;
        }

        public List<Node> getConnections(int level) {
            if (level < this.connections.size()) {
                return Collections.unmodifiableList(this.connections.get(level));
            }
            return Collections.emptyList();
        }

        public void addConnection(Node newNeighbour, int level) {
            List<Node> levelNeighbours = this.connections.get(level);

            levelNeighbours.add(newNeighbour);

            if (levelNeighbours.size() > getM(this.getParameters().getM(), level)) {
                this.connections.set(level, this.selectBestForConnecting(levelNeighbours));
            }
        }

        public abstract List<Node> selectBestForConnecting(List<Node> candidates);

        protected int getM(int baseM, int level) {
            return level == 0 ? 2 * baseM : baseM;
        }

    }


    private class NodeAlg3 extends Node {

        NodeAlg3(int id, TItem item, int maxLevel, BiFunction<TItem, TItem, TDistance> distance, Parameters parameters) {
            super(id, item, maxLevel, distance, parameters);
        }

        @Override
        public List<Node> selectBestForConnecting(List<Node> candidates) {

            Comparator<Node> fartherIsLess = this.getTravelingCosts().reversed();

            BinaryHeap<Node> candidatesHeap = new BinaryHeap<>(candidates, fartherIsLess);

            List<Node> result = new ArrayList<>(getM(this.getParameters().getM(), this.getMaxLevel()) + 1);

            while (!candidatesHeap.getBuffer().isEmpty() && result.size() < getM(this.getParameters().getM(), this.getMaxLevel())) {
                result.add(candidatesHeap.pop());
            }

            return result;
        }
    }


    private class NodeAlg4 extends Node {


        public NodeAlg4(int id, TItem item, int maxLevel, BiFunction<TItem, TItem, TDistance> distance, Parameters parameters) {
            super(id, item, maxLevel, distance, parameters);
        }

        @Override
        public List<Node> selectBestForConnecting(List<Node> candidates) {

            Comparator<Node> closerIsLess = this.getTravelingCosts();
            Comparator<Node> fartherIsLess = closerIsLess.reversed();

            BinaryHeap<Node> resultHeap = new BinaryHeap<>(
                    new ArrayList<>(getM(this.getParameters().getM(), this.getMaxLevel()) + 1), closerIsLess);

            BinaryHeap<Node> candidatesHeap = new BinaryHeap<>(candidates, fartherIsLess);


            if (this.getParameters().isExpandBestSelection()) {
                Set<Integer> candidatesIds = candidates.stream().map(Node::getId).collect(Collectors.toSet());

                for (Node neighbour : this.getConnections(this.getMaxLevel())) {
                    if (!candidatesIds.contains(neighbour.getId())) {
                        candidatesHeap.push(neighbour);
                        candidatesIds.add(neighbour.getId());
                    }
                }
            }

            BinaryHeap<Node> discardedHeap = new BinaryHeap<>(new ArrayList<>(candidatesHeap.getBuffer().size()), fartherIsLess);

            while (!candidatesHeap.getBuffer().isEmpty() && resultHeap.getBuffer().size() < getM(getParameters().getM(), this.getMaxLevel())) {
                Node candidate = candidatesHeap.pop();

                Optional<Node> farestResult = resultHeap.getBuffer().stream().findFirst();

                if (!farestResult.isPresent() ||
                        dLt(this.getTravelingCosts().from(candidate), this.getTravelingCosts().from(farestResult.get()))) {

                    resultHeap.push(candidate);
                } else if (this.getParameters().isKeepPrunedConnections()) {
                    discardedHeap.push(candidate);
                }
            }

            if (this.getParameters().isKeepPrunedConnections()) {

                while (!discardedHeap.getBuffer().isEmpty() && resultHeap.getBuffer().size() < getM(this.getParameters().getM(), this.getMaxLevel())) {
                    resultHeap.push(discardedHeap.pop());
                }
            }

            return resultHeap.getBuffer();

        }
    }


    class Graph {

        private final Parameters parameters;

        private TriFunction<Integer, TItem, Integer, Node> newNode;

        private Node entryPoint;

        public Graph(BiFunction<TItem, TItem, TDistance> distance, Parameters parameters) {
            this.parameters = parameters;

            switch (this.parameters.neighbourHeuristic) {
                case SELECTION_HEURISTIC:
                    this.newNode = (Integer id, TItem item, Integer level) -> new NodeAlg4(id, item, level, distance, this.parameters);
                    break;

                case SELECT_SIMPLE:
                default:
                    this.newNode = (Integer id, TItem item, Integer level) -> new NodeAlg3(id, item, level, distance, this.parameters);
                    break;
            }


        }

        public Parameters getParameters() {
            return parameters;
        }

        public TriFunction<Integer, TItem, Integer, Node> getNewNode() {
            return newNode;
        }


        public void create(List<TItem> items, DotNetRandom generator) {

            if (items == null || items.isEmpty()) {
                return;
            }

            int id = 0;
            Node entryPoint = this.newNode.apply(id, items.get(id), randomLevel(generator, this.getParameters().getLevelLambda()));

            for (id = 1; id < items.size(); id++) {

                // zoom in and find the best peer on the same level as newNode
                Node bestPeer = entryPoint;
                Node newNode = getNewNode().apply(id, items.get(id), randomLevel(generator, this.getParameters().getLevelLambda()));
                for (int level = bestPeer.getMaxLevel(); level > newNode.getMaxLevel(); --level) {
                    bestPeer = kNearestAtLevel(bestPeer, newNode, 1, level).get(0);
                }

                // connecting new node to the small world
                for (int level = Math.min(newNode.getMaxLevel(), entryPoint.getMaxLevel()); level >= 0; --level) {
                    List<Node> potentialNeighbours = kNearestAtLevel(bestPeer, newNode, this.getParameters().getConstructionPruning(), level);
                    List<Node> bestNeighbours = newNode.selectBestForConnecting(potentialNeighbours);

                    for (Node newNeighbour : bestNeighbours) {
                        newNode.addConnection(newNeighbour, level);
                        newNeighbour.addConnection(newNode, level);

                        // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                        if (dLt(newNode.getTravelingCosts().from(newNeighbour), newNode.getTravelingCosts().from(bestPeer))) {
                            bestPeer = newNeighbour;
                        }
                    }
                }

                // zoom out to the highest level
                if (newNode.getMaxLevel() > entryPoint.getMaxLevel()) {
                    entryPoint = newNode;
                }
            }

            // construction is done
            this.entryPoint = entryPoint;
        }


        /// <summary>
        /// Get k nearest items for a given one.
        /// Contains implementation of K-NN-SEARCH(hnsw, q, K, ef) algorithm.
        /// Article: Section 4. Algorithm 5.
        /// </summary>
        /// <param name="destination">The given node to get the nearest neighbourhood for.</param>
        /// <param name="k">The size of the neighbourhood.</param>
        /// <returns>The list of the nearest neighbours.</returns>
        public List<Node> kNearest(Node destination, int k) {
            Node bestPeer = this.entryPoint;
            for (int level = this.entryPoint.getMaxLevel(); level > 0; --level) {
                bestPeer = kNearestAtLevel(bestPeer, destination, 1, level).get(0);
            }

            return kNearestAtLevel(bestPeer, destination, k, 0);
        }


        /// <summary>
        /// The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
        /// Article: Section 4. Algorithm 2.
        /// </summary>
        /// <param name="entryPoint">The entry point for the search.</param>
        /// <param name="destination">The search target.</param>
        /// <param name="k">The number of the nearest neighbours to get from the layer.</param>
        /// <param name="level">Level of the layer.</param>
        /// <returns>The list of the nearest neighbours at the level.</returns>
        private List<Node> kNearestAtLevel(Node entryPoint, Node destination, int k, int level) {
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
            Comparator<Node> closerIsLess = destination.getTravelingCosts();
            Comparator<Node> fartherIsLess = closerIsLess.reversed();

            // prepare heaps

            List<Node> resultHeapValues = new ArrayList<>(k + 1);
            resultHeapValues.add(entryPoint);
            BinaryHeap<Node> resultHeap = new BinaryHeap<>(resultHeapValues, closerIsLess);


            List<Node> expansionHeapValues = new ArrayList<>();
            expansionHeapValues.add(entryPoint);
            BinaryHeap<Node> expansionHeap = new BinaryHeap<>(expansionHeapValues, fartherIsLess);

            // run bfs
            Set<Integer> visited = new HashSet<>();
            visited.add(entryPoint.getId());

            while (!expansionHeap.getBuffer().isEmpty()) {
                // get next candidate to check and expand
                Node toExpand = expansionHeap.pop();
                Node farthestResult = resultHeap.getBuffer().get(0);

                if (dGt(destination.getTravelingCosts().from(toExpand), destination.getTravelingCosts().from(farthestResult))) {
                    // the closest candidate is farther than farthest result
                    break;
                }

                // expand candidate
                for (Node neighbour : toExpand.getConnections(level)) {

                    if (!visited.contains(neighbour.getId())) {
                        // enque perspective neighbours to expansion list
                        farthestResult = resultHeap.getBuffer().get(0);
                        if (resultHeap.getBuffer().size() < k
                                || dLt(destination.getTravelingCosts().from(neighbour), destination.getTravelingCosts().from(farthestResult))) {
                            expansionHeap.push(neighbour);
                            resultHeap.push(neighbour);
                            if (resultHeap.getBuffer().size() > k) {
                                resultHeap.pop();
                            }
                        }

                        // update visited list
                        visited.add(neighbour.getId());
                    }
                }
            }

            return resultHeap.getBuffer();
        }

        /// <summary>
        /// Serializes edges of the graph.
        /// </summary>
        /// <returns>Bytes representing edges.</returns>
        public byte[] serialize() {

            throw new UnsupportedOperationException("implement");

//            using (var stream = new MemoryStream())
//            {
//                var formatter = new BinaryFormatter();
//                formatter.Serialize(stream, this.entryPoint.Id);
//                formatter.Serialize(stream, this.entryPoint.MaxLevel);
//
//                for (int level = this.entryPoint.MaxLevel; level >= 0; --level)
//                {
//                    var edges = new Dictionary<int, List<int>>();
//                    BFS(this.entryPoint, level, (node) =>
//                    {
//                        edges[node.Id] = node.GetConnections(level).Select(x => x.Id).ToList();
//                    });
//
//                    formatter.Serialize(stream, edges);
//                }
//
//                return stream.ToArray();
//            }
        }

        /// <summary>
        /// Deserilaizes graph edges and assigns nodes to the items.
        /// </summary>
        /// <param name="items">The underlying items.</param>
        /// <param name="bytes">The serialized edges.</param>
        public void deserialize(List<TItem> items, byte[] bytes) {

            throw new UnsupportedOperationException("implement");

//            var nodeList = Enumerable.Repeat<Node>(null, items.Count).ToList();
//            Func<int, int, Node> getOrAdd = (id, level) => nodeList[id] = nodeList[id] ?? this.NewNode(id, items[id], level);
//
//            using (var stream = new MemoryStream(bytes))
//            {
//                var formatter = new BinaryFormatter();
//                int entryId = (int)formatter.Deserialize(stream);
//                int maxLevel = (int)formatter.Deserialize(stream);
//
//                nodeList[entryId] = this.NewNode(entryId, items[entryId], maxLevel);
//                for (int level = maxLevel; level >= 0; --level)
//                {
//                    var edges = (Dictionary<int, List<int>>)formatter.Deserialize(stream);
//                    foreach (var pair in edges)
//                    {
//                        var currentNode = getOrAdd(pair.Key, level);
//                        foreach (var adjacentId in pair.Value)
//                        {
//                            var neighbour = getOrAdd(adjacentId, level);
//                            currentNode.AddConnection(neighbour, level);
//                        }
//                    }
//                }
//
//                this.entryPoint = nodeList[entryId];
//            }
        }

        /// <summary>
        /// Prints edges of the graph.
        /// </summary>
        /// <returns>String representation of the graph's edges.</returns>
        String print() {
            StringBuilder buffer = new StringBuilder();
            for (int level = this.entryPoint.getMaxLevel(); level >= 0; --level) {
                buffer.append(String.format("[LEVEL %s]%n", level));
                int finalLevel = level;
                bfs(this.entryPoint, level, node -> {
                    String neighbours = node.getConnections(finalLevel).stream().map(x -> String.valueOf(x.getId()))
                            .collect(Collectors.joining(","));
                    buffer.append(String.format("(%d) -> {%s}%n", node.getId(), neighbours));
                });
                buffer.append(String.format("%n"));
            }

            return buffer.toString();
        }

        private int randomLevel(DotNetRandom generator, double lambda) {
            double r = -Math.log(generator.nextDouble()) * lambda;
            return (int) r;
        }
    }


}

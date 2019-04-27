package org.github.jelmerk.hnsw;

import java.io.Serializable;
import java.util.*;

/**
 * The implementation of the node in hnsw graph.
 */
class Node implements Serializable {

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

    /**
     * The abstract class representing algorithm to control node capacity.
     *
     * @param <TItem> The typeof the items in the small world.
     * @param <TDistance> The type of the distance in the small world.
     */
    static abstract class Algorithm<TItem, TDistance extends Comparable<TDistance>> implements Serializable {

        // Gives access to the core of the graph.
        protected Graph<TItem, TDistance>.Core graphCore;

        /// Cache of the distance function between the nodes.
        protected DistanceFunction<Integer, TDistance> nodeDistance;

        /**
         * Initializes a new instance of the {@link Algorithm} class
         *
         * @param graphCore The core of the graph.
         */
        public Algorithm(Graph<TItem, TDistance>.Core graphCore) {
            this.graphCore = graphCore;
            this.nodeDistance = graphCore::calculateDistance;
        }

        /**
         * Creates a new instance of the {@link Node} struct. Controls the exact type of connection lists.
         *
         * @param nodeId The identifier of the node.
         * @param maxLayer The max layer where the node is presented.
         * @return The new instance.
         */
        protected Node newNode(int nodeId, int maxLayer) {
            List<List<Integer>> connections = new ArrayList<>(maxLayer + 1);
            for (int layer = 0; layer <= maxLayer; layer++) {
                // M + 1 neighbours to not realloc in addConnection when the level is full
                int layerM = this.getM(layer) + 1;
                connections.add(new ArrayList<>(layerM));
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
            return layer == 0 ? 2 * this.graphCore.getParameters().getM() : this.graphCore.getParameters().getM();
        }

        /**
         * Tries to connect the node with the new neighbour.
         *
         * @param node The node to add neighbour to.
         * @param neighbour The new neighbour.
         * @param layer The layer to add neighbour to.
         */
        void connect(Node node, Node neighbour, int layer) {
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
     *
     * @param <TItem> The typeof the items in the small world.
     * @param <TDistance> The type of the distance in the small world.
     */
    static class Algorithm3<TItem, TDistance extends Comparable<TDistance>> extends Algorithm<TItem, TDistance> {

        /**
         * Initializes a new instance of the {@link Algorithm3} class.
         *
         * @param graphCore The core of the graph.
         */
        public Algorithm3(Graph<TItem, TDistance>.Core graphCore) {
            super(graphCore);
        }

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
     *
     * @param <TItem> The typeof the items in the small world.
     * @param <TDistance> The type of the distance in the small world.
     */
    static class Algorithm4<TItem, TDistance extends Comparable<TDistance>> extends Algorithm<TItem, TDistance> {

        /**
         * Initializes a new instance of the {@link Algorithm4} class.
         *
         * @param graphCore The core of the graph.
         */
        public Algorithm4(Graph<TItem, TDistance>.Core graphCore) {
            super(graphCore);
        }

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
            if (this.graphCore.getParameters().isExpandBestSelection()) {

                Set<Integer> visited = new HashSet<>(candidatesHeap.getBuffer());

                for (Integer candidateId: candidatesHeap.getBuffer()) {

                    for(Integer candidateNeighbourId : this.graphCore.getNodes().get(candidateId).getConnections(layer)) {

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

                }  else if (this.graphCore.getParameters().isKeepPrunedConnections()) {
                    discardedHeap.push(candidateId);
                }
            }

            // keep pruned option is enabled
            if (this.graphCore.getParameters().isKeepPrunedConnections()) {
                while (!discardedHeap.getBuffer().isEmpty() && resultHeap.getBuffer().size() < layerM) {
                    resultHeap.push(discardedHeap.pop());
                }
            }

            return resultHeap.getBuffer();
        }
    }


}

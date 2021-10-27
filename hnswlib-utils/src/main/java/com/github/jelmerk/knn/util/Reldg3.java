package com.github.jelmerk.knn.util;

import org.eclipse.collections.api.map.primitive.IntIntMap;
import org.eclipse.collections.api.map.primitive.MutableIntIntMap;
import org.eclipse.collections.api.map.primitive.MutableIntObjectMap;
import org.eclipse.collections.impl.map.mutable.primitive.IntIntHashMap;
import org.eclipse.collections.impl.map.mutable.primitive.IntObjectHashMap;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Reldg3 {

    interface Node {

        int id();

        int[] edges();
    }

    static class SimpleNode implements Node {

        private final int id;
        private final int[] edges;

        public SimpleNode(int id, int[] edges) {
            this.id = id;
            this.edges = edges;
        }

        @Override
        public int id() {
            return id;
        }

        @Override
        public int[] edges() {
            return edges;
        }

        @Override
        public String toString() {
            return "SimpleNode{" +
                    "id=" + id +
                    ", edges=" + Arrays.toString(edges) +
                    '}';
        }
    }


    

    public static Node[] read(InputStream inputStream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));

        MutableIntObjectMap<SimpleNode> map = new IntObjectHashMap<>();

        String line;
        while((line = reader.readLine()) != null) {
            String[] tokens = line.split(" ");

            int from = Double.valueOf(tokens[0]).intValue();
            int to = Double.valueOf(tokens[1]).intValue();

//            if (from == 12129 || to == 12129) {
//                System.out.println("## " + line);
//            }

            SimpleNode node = map.get(from);

            if (node == null) {
                node = new SimpleNode(from, new int[] { to });
            } else {
                int[] edges = new int[node.edges.length + 1];
                System.arraycopy(node.edges,0, edges, 0, node.edges.length);
                edges[node.edges.length] = to;
                node = new SimpleNode(from, edges);
            }
            map.put(from, node);
        }

        return map.values().toArray(new Node[] {});
    }





    /**
     * Sorts by the number of edges highest number of edges first
     */
    static class NodeDegreesComparator implements Comparator<Node> {

        public static NodeDegreesComparator INSTANCE = new NodeDegreesComparator();

        private NodeDegreesComparator() {
        }

        @Override
        public int compare(Node node, Node other) {
            return Integer.compare(other.edges().length, node.edges().length);
        }

    }


    static List<Node> bfsDisconnected(Node[] nodes) {

        // TODO maybe just change the type of the edges from int to node ?? otherwise we keep having to construct these maps

        MutableIntObjectMap<Node> map = new IntObjectHashMap<>(nodes.length);
        for (Node node : nodes) {
            map.put(node.id(), node);
        }

        Arrays.sort(nodes, NodeDegreesComparator.INSTANCE); // TODO nasty, sorts in place

        Set<Node> visited = new HashSet<>();
        
        List<Node> reordered = new ArrayList<>(nodes.length);

        for (Node current : nodes) {

            if (visited.contains(current)) {
                continue;
            }

            Queue<Node> queue = new ArrayDeque<>();
            queue.add(current);

            while (!queue.isEmpty()) {
                Node node = queue.poll();

                if (visited.contains(node)) {
                    continue;
                }

                reordered.add(node);

                visited.add(node);

                for (int edgeId : node.edges()) {
                    queue.add(map.get(edgeId));
                }
            }
        }


        return reordered;

    }


    /**
     * This algorithm favors a cluster if it has many neighbors of a node, but
     * penalizes the cluster if it is close to capacity.
     *
     * edges: An [:,2] array of edges.
     * stream_order_indices: A list of indices in which to stream over the edges.
     * num_nodes: The number of nodes in the graph.
     * num_partitions: How many partitions we are breaking the graph into.
     * partition: The partition from a previous run. Used for restreaming.
     *
     * Returns: A new partition.
     */


    private static final int UNDEFINED = -1;



    public static double shardmapEvaluate(int[] shardMap, List<Node> nodes) {

        long internal = 0;
        long total = 0;

        for (Node node : nodes) {

            int nodePartition = shardMap[node.id()];

            internal += Arrays.stream(node.edges()).map(edgeId -> shardMap[edgeId] == nodePartition ? 1 : 0).sum();
            total += node.edges().length;
        }


        return (double) internal / (double) total;

    }

    public static IntIntMap distributionEvaluate(int[] shardMap) {

        MutableIntIntMap map = new IntIntHashMap();

        for (int shard : shardMap) {
            int current = map.getIfAbsent(shard, 0);
            map.put(shard, current + 1);
        }
        return map;
    }




    // TODO work out return type

    public static int[] reldg(List<Node> nodes,
                              int numPartitions,
                              int numIterations,
                              double epsilon) {

        int[] partition = new int[nodes.size()];
        Arrays.fill(partition, UNDEFINED);

        for (int iteration = 0; iteration < numIterations; iteration++) {

            System.out.println("iteration " + iteration);

            int[] partitionSizes = new int[numPartitions];
            int[] partitionVotes = new int[numPartitions];

            int partitionCapacity = (int) Math.ceil (((double)nodes.size() / (double) numPartitions) * (1.0 + epsilon));

            for (Node node : nodes) {

                // if any of the edge nodes is already assigned to a partition then that node
                // casts a vote to be assigned to that partition

                for (int edge : node.edges()) {

                    int edgePartition = partition[edge];

                    if (edgePartition != UNDEFINED) {
                        partitionVotes[edgePartition] += 1;
                    }
                }


                // based on the cast votes assign the node to a partition
                int selectedPartition = UNDEFINED;
                int maxVal = 0;

                for (int partitionNum = 0; partitionNum < numPartitions; partitionNum++) {

                    int spaceLeftInPartition = partitionCapacity - partitionSizes[partitionNum];
                    int val = partitionVotes[partitionNum] * spaceLeftInPartition;


                    if (val > maxVal) {
                        selectedPartition = partitionNum;
                        maxVal = val;
                    }
                }


                // TODO: all this is just tiebreaker code

                if (maxVal == 0) {

                    for (int currentPartition = node.id() % numPartitions; currentPartition < numPartitions;  currentPartition++) {
                        if (partitionSizes[currentPartition] < partitionCapacity) {
                            selectedPartition = currentPartition;
                            maxVal = 1;
                            break;
                        }
                    }

                    if (maxVal == 0) {
                        for (int currentPartition = 0; currentPartition < node.id() %  numPartitions; currentPartition++) {
                            if (partitionSizes[currentPartition] < partitionCapacity) {
                                selectedPartition = currentPartition;
                                break;
                            }
                        }
                    }

                }

                partitionSizes[selectedPartition] += 1;
                partition[node.id()] = selectedPartition;
                Arrays.fill(partitionVotes, 0);

            }


            System.out.println(shardmapEvaluate(partition, nodes));

            System.out.println(distributionEvaluate(partition));
        }

        return partition;

    }


    public static void main(String[] args) throws Exception {
//        Node[] nodes = read(new FileInputStream("/home/jkuperus/dev/3rdparty/streamorder/data/web-NotreDame_edges.txt"));
//
//
//        System.out.println(Arrays.stream(nodes).filter(v -> v.id() == 12129).findFirst().get().edges().length);
//
//        System.out.println(
//            Arrays.stream(nodes).mapToInt(v -> v.edges().length).sum()
//        );
//
//
//        List<Node> reordered = bfsDisconnected(nodes);
//
//        System.out.println(
//                reordered.stream().mapToInt(v -> v.edges().length).sum()
//        );
//
//
//
//        reldg(reordered, 16, 10, 0.0);


//        reordered.stream().mapToInt(Node::id).limit(1000).forEach(i -> System.out.println(i));
//        System.out.println(nodes.length);

        List<String> lines = Files.readAllLines(Paths.get("/home/jkuperus/reldg-node-weight-connections.csv"));

        Reldg3.SimpleNode[] nodes = lines.stream().map(line -> {
            String[] tokens = line.split(",");
            int[] fields = Arrays.stream(tokens).mapToInt(Integer::valueOf).toArray();

            int[] connections = new int[fields.length - 2];
            System.arraycopy(fields, 2, connections, 0, connections.length);
            return new Reldg3.SimpleNode(fields[0], connections);
        }).toArray(Reldg3.SimpleNode[]::new);


        List<Reldg3.Node> reordered = bfsDisconnected(nodes);

        reldg(reordered, 4, 100, 0.0);

    }

}

package org.github.jelmerk.hnsw;

import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class SmallWorldTest {

    private float floatError = 0.000000596f;

    private List<float[]> vectors;

    @Before
    public void setUp() throws Exception {
        this.vectors = readTextFile("/vectors.txt").stream()
                .map(l -> {
                        String[] tokens = l.split("\t");

                        float[] floats = new float[tokens.length];

                        for (int i = 0; i < tokens.length; i++) {
                            floats[i] = Float.parseFloat(tokens[i]);
                        }

                        return floats;
                })
                .collect(Collectors.toList());
    }

    @Test
    public void testKnnSearch() {

        SmallWorld.Parameters parameters = new SmallWorld.Parameters();
        SmallWorld<float[]> graph = new SmallWorld<>(CosineDistance::nonOptimized);
        graph.buildGraph(this.vectors, new DotNetRandom(42), parameters);

        for (int i = 0; i < this.vectors.size(); i++) {

            List<SmallWorld.KNNSearchResult<float[]>> result = graph.knnSearch(this.vectors.get(i), 20);
            result.sort(Comparator.comparing(SmallWorld.KNNSearchResult::getDistance));

            SmallWorld.KNNSearchResult<float[]> best = result.get(0);

            assertEquals(20, result.size());
            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

    }

    @Test
    public void testSerialization() throws Exception {
        SmallWorld.Parameters parameters = new SmallWorld.Parameters();
        SmallWorld<float[]> graph = new SmallWorld<>(CosineDistance::nonOptimized);
        graph.buildGraph(this.vectors, new DotNetRandom(42), parameters);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);

        oos.writeObject(graph);
    }

    private List<String> readTextFile(String path) throws IOException {
        try(BufferedReader reader = new BufferedReader(new InputStreamReader(getClass().getResourceAsStream(path)))) {
            String line;

            List<String> result = new ArrayList<>();
            while((line = reader.readLine()) != null) {
                result.add(line);
            }
            return result;
        }
    }
}

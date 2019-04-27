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
        SmallWorld<float[], Float> graph = new SmallWorld<>(CosineDistance::nonOptimized);
        graph.buildGraph(this.vectors, new DotNetRandom(42), parameters);

        for (int i = 0; i < this.vectors.size(); i++) {

            List<SmallWorld.KNNSearchResult<float[], Float>> result = graph.knnSearch(this.vectors.get(i), 20);
            result.sort(Comparator.comparing(SmallWorld.KNNSearchResult::getDistance));

            SmallWorld.KNNSearchResult<float[], Float> best = result.get(0);

            assertEquals(20, result.size());
            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

    }

    @Test
    public void testSerialization() throws Exception {
        SmallWorld.Parameters parameters = new SmallWorld.Parameters();
        SmallWorld<float[], Float> original = new SmallWorld<>(CosineDistance::nonOptimized);
        original.buildGraph(this.vectors, new DotNetRandom(42), parameters);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        original.save(baos);

        SmallWorld<float[], Float> loaded = SmallWorld.load(new ByteArrayInputStream(baos.toByteArray()));

        assertEquals(original.print(), loaded.print());
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

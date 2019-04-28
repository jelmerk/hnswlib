package org.github.jelmerk.hnsw;

import org.github.jelmerk.SearchResult;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class HnswAlgoritmTest {

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

        DotNetRandom random = new DotNetRandom(42);
        Parameters parameters = new Parameters();
        HnswAlgorithm<float[], Float> graph = new HnswAlgorithm<>(random, parameters, CosineDistance::nonOptimized);

        for (float[] vector : vectors) {
            graph.addItem(vector);
        }

        for (int i = 0; i < this.vectors.size(); i++) {

            List<SearchResult<float[], Float>> result = graph.search(this.vectors.get(i), 20);
            result.sort(Comparator.comparing(SearchResult::getDistance));

            SearchResult<float[], Float> best = result.get(0);

            assertEquals(20, result.size());
//            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

    }

    @Test
    public void testSerialization() throws Exception {
        DotNetRandom random = new DotNetRandom(42);
        Parameters parameters = new Parameters();
        HnswAlgorithm<float[], Float> original = new HnswAlgorithm<>(random, parameters, CosineDistance::nonOptimized);
        for (float[] vector : vectors) {
            original.addItem(vector);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        original.saveIndex(baos);

        HnswAlgorithm<float[], Float> loaded = HnswAlgorithm.load(new ByteArrayInputStream(baos.toByteArray()));

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

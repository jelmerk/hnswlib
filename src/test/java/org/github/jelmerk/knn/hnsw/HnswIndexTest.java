package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.DistanceFunctions;
import org.github.jelmerk.knn.SearchResult;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class HnswIndexTest {

    private float floatError = 0.000000596f;

    private List<TestItem> items;

    @Before
    public void setUp() throws Exception {
        this.items = readTextFile("/vectors.txt").stream()
                .map(l -> {
                        String[] tokens = l.split("\t");

                        float[] floats = new float[tokens.length];

                        for (int i = 0; i < tokens.length; i++) {
                            floats[i] = Float.parseFloat(tokens[i]);
                        }

                        return new TestItem(UUID.randomUUID().toString(), floats);
                })
                .collect(Collectors.toList());
    }


    @Test
    public void testKnnSearch() {

        HnswIndex<String, float[], TestItem, Float> index =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, items.size())
                        .build();

        for (TestItem item : items) {
            index.add(item);
        }

        for (TestItem item : this.items) {

            List<SearchResult<TestItem, Float>> result = index.findNearest(item.getVector(), 20);

            SearchResult<TestItem, Float> best = result.iterator().next();

            assertEquals(20, result.size());
            assertEquals(0, best.getDistance(), floatError);
        }

    }

    @Test
    public void testSerialization() throws Exception {

        HnswIndex<String, float[], TestItem, Float> original =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, items.size())
                        .build();

        for (TestItem item : items) {
            original.add(item);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        original.save(baos);

        HnswIndex<String, float[], TestItem, Float> loaded = HnswIndex.load(new ByteArrayInputStream(baos.toByteArray()));

//        assertEquals(original.print(), loaded.print());
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

package org.github.jelmerk.hnsw;

import org.github.jelmerk.SearchResult;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
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

        DotNetRandom random = new DotNetRandom(42);
        Parameters parameters = new Parameters();
        parameters.setMaxItemCount(items.size());

        HnswIndex<String, float[], TestItem, Float> index = new HnswIndex<>(random, parameters, CosineDistance::nonOptimized);

        for (TestItem item : items) {
            index.add(item);
        }

        for (int i = 0; i < this.items.size(); i++) {

            List<SearchResult<TestItem, Float>> result = index.findNearest(this.items.get(i).getVector(), 20);
            result.sort(Comparator.comparing(SearchResult::getDistance));

            SearchResult<TestItem, Float> best = result.get(0);

            assertEquals(20, result.size());
//            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

        System.out.println(index.print());


    }

    @Test
    public void testSerialization() throws Exception {
        DotNetRandom random = new DotNetRandom(42);
        Parameters parameters = new Parameters();
        parameters.setMaxItemCount(items.size());
        HnswIndex<String, float[], TestItem, Float> original = new HnswIndex<>(random, parameters, CosineDistance::nonOptimized);
        for (TestItem item : items) {
            original.add(item);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        original.save(baos);

        HnswIndex<String, float[], TestItem, Float> loaded = HnswIndex.load(new ByteArrayInputStream(baos.toByteArray()));

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

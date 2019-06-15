package com.github.jelmerk.knn.hnsw;

import com.github.jelmerk.knn.FloatDistanceFunctions;
import com.github.jelmerk.knn.SearchResult;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;
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
    public void testReplace() {

        ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

        ReentrantReadWriteLock.WriteLock writeLock = lock.writeLock();


        writeLock.lock();
        writeLock.lock();


        HnswIndex<String, float[], TestItem, Float> index = HnswIndex
                .newBuilder(FloatDistanceFunctions::cosineDistance, 1)
                .withRemoveEnabled()
                .build();

        index.add(items.get(0));
        index.add(items.get(0));
    }


    @Test
    public void testKnnSearch() throws Exception{

        HnswIndex<String, float[], TestItem, Float> index = HnswIndex
                .newBuilder(FloatDistanceFunctions::cosineDistance, items.size())
                    .withRemoveEnabled()
                    .build();



        for (TestItem item : items) {
            index.add(item);
            index.add(item);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        index.save(baos);
        index = HnswIndex.load(new ByteArrayInputStream(baos.toByteArray()));

        for (TestItem item : this.items) {

            List<SearchResult<TestItem, Float>> result = index.findNearest(item.vector(), 20);

            SearchResult<TestItem, Float> best = result.iterator().next();

            assertEquals(20, result.size());
            assertEquals(0, best.distance(), floatError);
        }

    }

    @Test
    public void testSerialization() throws Exception {

        ObjectSerializer<String> itemIdSerializer = new JavaObjectSerializer<>();
        ObjectSerializer<TestItem> itemSerializer = new JavaObjectSerializer<>();

        HnswIndex<String, float[], TestItem, Float> original = HnswIndex
                .newBuilder(FloatDistanceFunctions::cosineDistance, items.size())
                    .withCustomSerializers(itemIdSerializer, itemSerializer)
                    .build();

        System.out.println(items.size());

        for (TestItem item : items) {
            original.add(item);
        }

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        original.save(baos);
        baos.flush();

        System.out.println(baos.toByteArray().length); // 129280
                                                       // 129075

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

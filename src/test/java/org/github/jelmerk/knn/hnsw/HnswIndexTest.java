package org.github.jelmerk.knn.hnsw;

import org.github.jelmerk.knn.DistanceFunctions;
import org.github.jelmerk.knn.SearchResult;
import org.junit.Before;
import org.junit.Test;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
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
    public void findJavaSeedWhichWorksIdenticallyForTest() throws Exception {

        DotNetRandom dotNetRandom = new DotNetRandom(42);

        double lambda = 1 / Math.log(10);

        final List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < 225; i++) {
            expected.add(randomLayer(dotNetRandom, lambda));
        }

        int numProcessors = Runtime.getRuntime().availableProcessors();

        AtomicLong seed = new AtomicLong();

        ExecutorService executorService = Executors.newFixedThreadPool(numProcessors);


        for (int i = 0; i < numProcessors; i++) {
            executorService.submit(() -> {
                List<Integer> result = new ArrayList<>();

                long seedValue;

                do {

                    result.clear();

                    seedValue = seed.getAndIncrement();

                    Random random = new Random(seedValue);

                    for (int j = 0; j < 225; j++) {
                        result.add(randomLayer(random, lambda));
                    }
                } while(!expected.equals(result));

                System.out.println("Got one : " + seedValue);

            });
        }

        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.DAYS);

    }



    int randomLayer(DotNetRandom generator, double lambda) {
        double r = -Math.log(generator.nextDouble()) * lambda;
        return (int)r;
    }

    int randomLayer(Random generator, double lambda) {
        double r = -Math.log(generator.nextDouble()) * lambda;
        return (int)r;
    }

    @Test
    public void testKnnSearch() {

        HnswIndex<String, float[], TestItem, Float> index =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, items.size())
//                        .setNeighbourHeuristic(NeighbourSelectionHeuristic.SELECT_HEURISTIC)
                        .setRandomSeed(42)
                        .build();

        for (TestItem item : items) {
            index.add(item);
        }


//        System.out.println(index.print());


//        TestItem item = items.get(1);
////        TestItem item = items.get(1);
//        List<SearchResult<TestItem, Float>> result = index.findNearest2(item.getVector(), 20);
////        List<SearchResult<TestItem, Float>> result = index.findNearest(item.getVector(), 20);
//        SearchResult<TestItem, Float> best = result.iterator().next();
//
//
////        assertEquals(20, result.size()); // TODO i think this may be a bug , node 0 has 2 relations to itself which makes no sense
//
//        assertEquals(0, best.getDistance(), floatError);

        for (TestItem item : this.items) {

            List<SearchResult<TestItem, Float>> result = index.findNearest(item.getVector(), 20);
//            List<SearchResult<TestItem, Float>> result = index.findNearest2(item.getVector(), 20);

            SearchResult<TestItem, Float> best = result.iterator().next();

            assertEquals(20, result.size());
//            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

//        System.out.println(index.print());





    }

    @Test
    public void testSerialization() throws Exception {

        HnswIndex<String, float[], TestItem, Float> original =
                new HnswIndex.Builder<>(DistanceFunctions::cosineDistance, items.size())
                        .setRandomSeed(42)
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

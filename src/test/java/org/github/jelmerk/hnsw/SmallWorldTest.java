package org.github.jelmerk.hnsw;

import org.junit.Before;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class SmallWorldTest {

    private float floatError = 0.000000596f;

    private List<Float[]> vectors;

    @Before
    public void setUp() throws Exception {
        this.vectors = readTextFile("/vectors.txt").stream()
                .map(l -> Arrays.stream(l.split("\t"))
                            .map(Float::parseFloat)
                            .toArray(Float[]::new))
                .collect(Collectors.toList());
    }

    @Test
    public void testKnnSearch() {
        Random random = new Random(42);

        // TODO: work out how to not use boxed primitives here

        SmallWorld.Parameters parameters = new SmallWorld.Parameters();
        parameters.setM(15);
        parameters.setLevelLambda(1 / Math.log(parameters.getM()));

        SmallWorld<Float[], Float> graph = new SmallWorld<>(CosineDistance::nonOptimized);
        graph.buildGraph(this.vectors, new DotNetRandom(42), parameters);

        System.out.println(graph.print());

        for (int i = 0; i < this.vectors.size(); ++i) {
            List<SmallWorld<Float[], Float>.KNNSearchResult> results = graph.kNNSearch(this.vectors.get(i), 20);

            SmallWorld<Float[], Float>.KNNSearchResult best = results.stream()
                    .sorted(Comparator.comparing((Function<SmallWorld<Float[], Float>.KNNSearchResult, Float>) SmallWorld.KNNSearchResult::getDistance))
                    .collect(Collectors.toList())
                    .get(0);

            assertEquals(i, best.getId());
            assertEquals(0, best.getDistance(), floatError);
        }

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

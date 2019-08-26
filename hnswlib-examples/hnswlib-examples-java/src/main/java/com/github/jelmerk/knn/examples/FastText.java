package com.github.jelmerk.knn.examples;

import com.github.jelmerk.knn.DistanceFunctions;
import com.github.jelmerk.knn.SearchResult;
import com.github.jelmerk.knn.hnsw.HnswIndex;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

import static com.github.jelmerk.knn.util.VectorUtils.normalize;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

/**
 * Example application that will download the english fast-text word vectors and insert them into a hnsw index.
 */
public class FastText {

    private static final String WORDS_FILE_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz";

    private static final Path TMP_PATH = Paths.get(System.getProperty("java.io.tmpdir"));

    public static void main(String[] args) throws Exception {

        Path file = TMP_PATH.resolve("cc.en.300.vec.gz");

        if (!Files.exists(file)) {
            downloadFile(WORDS_FILE_URL, file);
        } else {
            System.out.printf("Input file already downloaded. Using %s%n", file);
        }

        List<Word> words = loadWordVectors(file);

        System.out.println("Constructing index.");

        int m = 16;

        HnswIndex<String, float[], Word, Float> index = HnswIndex
                .newBuilder(DistanceFunctions.FLOAT_INNER_PRODUCT, words.size())
                .withM(m)
                .build();

        long start = System.currentTimeMillis();

        index.addAll(words, (workDone, max) -> System.out.printf("Added %d out of %d words to the index.%n", workDone, max));

        long end = System.currentTimeMillis();

        long duration = end - start;

        System.out.printf("Creating index with %d words took %d millis which is %d minutes.%n", index.size(), duration, MILLISECONDS.toMinutes(duration));

        List<SearchResult<Word, Float>> nearest = index.findNeighbors("bike", 10);

        for (SearchResult<Word, Float> result : nearest) {
            System.out.printf("%s %.4f%n", result.item().id(), result.distance());
        }
    }

    private static void downloadFile(String url, Path path) throws IOException {
        System.out.printf("Downloading %s to %s. This may take a while.%n", url, path);

        try (InputStream in = new URL(url).openStream()) {
            Files.copy(in, path);
        }
    }
    private static List<Word> loadWordVectors(Path path) throws IOException {
        System.out.printf("Loading words from %s%n", path);

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new GZIPInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8))) {
            return reader.lines()
                    .skip(1)
                    .map(line -> {
                        String[] tokens = line.split(" ");

                        String word = tokens[0];

                        float[] vector = new float[tokens.length - 1];
                        for (int i = 1; i < tokens.length - 1; i++) {
                            vector[i] = Float.valueOf(tokens[i]);
                        }

                        return new Word(word, normalize(vector)); // normalize the vector so we can do inner product search
                    })
                    .collect(Collectors.toList());
        }
    }

}

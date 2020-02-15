package com.github.jelmerk.knn.bruteforce;

import com.github.jelmerk.knn.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.assertThat;

class BruteForceIndexTest {

    private int dimensions = 2;
    private BruteForceIndex<String, float[], TestItem, Float> index;

    private TestItem item1 = new TestItem("1", new float[] { 0.0110f, 0.2341f }, 10);
    private TestItem item2 = new TestItem("2", new float[] { 0.2300f, 0.3891f }, 10);
    private TestItem item3 = new TestItem("3", new float[] { 0.4300f, 0.9891f }, 10);

    @BeforeEach
    void setUp() {
        index = BruteForceIndex
                    .newBuilder(dimensions, DistanceFunctions.FLOAT_COSINE_DISTANCE)
                    .build();
    }

    @Test
    void returnDimensions() {
        assertThat(index.getDimensions(), is(dimensions));
    }

    @Test
    void returnsSize() {
        assertThat(index.size(), is(0));
        index.add(item1);
        assertThat(index.size(), is(1));
    }

    @Test
    void addAndGet() {
        assertThat(index.get(item1.id()), is(Optional.empty()));
        index.add(item1);
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    void returnsItems() {
        assertThat(index.items().isEmpty(), is(true));
        index.add(item1);
        assertThat(index.items().size(), is(1));
        assertThat(index.items(), hasItems(item1));
    }

    @Test
    void removeItem() {
        index.add(item1);
        assertThat(index.remove(item1.id(), item1.version()), is(true));
        assertThat(index.size(), is(0));
        assertThat(index.items().size(), is(0));
        assertThat(index.get(item1.id()), is(Optional.empty()));
    }

    @Test
    void addNewerItem() {
        TestItem newerItem = new TestItem(item1.id(), new float[] {0f, 0f}, item1.version() + 1);

        index.add(item1);
        index.add(newerItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(newerItem)));
    }

    @Test
    void addOlderItem() {
        TestItem olderItem = new TestItem(item1.id(), new float[] {0f, 0f}, item1.version() - 1);

        index.add(item1);
        index.add(olderItem);

        assertThat(index.size(), is(1));
        assertThat(index.get(item1.id()), is(Optional.of(item1)));
    }

    @Test
    void removeUnknownItem() {
        assertThat(index.remove("foo", 0), is(false));
    }

    @Test
    void removeWithOldVersionIgnored() {
        index.add(item1);

        assertThat(index.remove(item1.id(), item1.version() - 1), is(false));
        assertThat(index.size(), is(1));
    }

    @Test
    void findNearest() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        List<SearchResult<TestItem, Float>> nearest = index.findNearest(item1.vector(), 10);

        assertThat(nearest, is(Arrays.asList(
            new SearchResult<>(item1, 0f, Comparator.naturalOrder()),
            new SearchResult<>(item3, 0.06521261f, Comparator.naturalOrder()),
            new SearchResult<>(item2, 0.11621308f, Comparator.naturalOrder())
        )));
    }

    @Test
    void findNeighbors() throws InterruptedException {
        index.addAll(Arrays.asList(item1, item2, item3));

        List<SearchResult<TestItem, Float>> nearest = index.findNeighbors(item1.id(), 10);

        assertThat(nearest, is(Arrays.asList(
                new SearchResult<>(item3, 0.06521261f, Comparator.naturalOrder()),
                new SearchResult<>(item2, 0.11621308f, Comparator.naturalOrder())
        )));
    }

    @Test
    void addAllCallsProgressListener() throws InterruptedException {
        List<ProgressUpdate> updates = new ArrayList<>();

        index.addAll(Arrays.asList(item1, item2, item3), 1,
                (workDone, max) -> updates.add(new ProgressUpdate(workDone, max)), 2);

        assertThat(updates, is(Arrays.asList(
            new ProgressUpdate(2, 3),
            new ProgressUpdate(3, 3)  // emitted because its the last element
        )));
    }

    @Test
    void saveAndLoadIndex() throws IOException {
        ByteArrayOutputStream in = new ByteArrayOutputStream();

        index.add(item1);

        index.save(in);

        BruteForceIndex<String, float[], TestItem, Float> loadedIndex =
                BruteForceIndex.load(new ByteArrayInputStream(in.toByteArray()));

        assertThat(loadedIndex.size(), is(1));
    }


}


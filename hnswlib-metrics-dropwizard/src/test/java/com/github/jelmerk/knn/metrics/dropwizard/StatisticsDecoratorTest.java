package com.github.jelmerk.knn.metrics.dropwizard;

import com.codahale.metrics.MetricRegistry;
import com.github.jelmerk.knn.Index;
import com.github.jelmerk.knn.SearchResult;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.*;

import static org.awaitility.Awaitility.*;
import static org.hamcrest.CoreMatchers.*;
import static org.mockito.BDDMockito.*;
import static com.codahale.metrics.MetricRegistry.name;

import static org.hamcrest.MatcherAssert.assertThat;

@ExtendWith(MockitoExtension.class)
class StatisticsDecoratorTest {

    @Mock
    private Index<String, float[], TestItem, Float> approximativeIndex;

    @Mock
    private Index<String, float[], TestItem, Float> groundTruthIndex;

    private final String indexName = "test_index";

    private MetricRegistry metricRegistry;

    private final TestItem item1 = new TestItem("1", new float[0]);
    private final TestItem item2 = new TestItem("2", new float[0]);

    private final int k = 10;

    private StatisticsDecorator<String, float[], TestItem, Float, Index<String, float[], TestItem, Float>, Index<String, float[], TestItem, Float>> decorator;

    @BeforeEach
    void setUp() {
        int maxAccuracySampleFrequency = 1;

        this.metricRegistry = new MetricRegistry();
        this.decorator = new StatisticsDecorator<>(metricRegistry, StatisticsDecoratorTest.class,
                indexName, approximativeIndex, groundTruthIndex, maxAccuracySampleFrequency);
    }

    @Test
    void timesAdd() {
        decorator.add(item1);
        verify(approximativeIndex).add(item1);
        assertThat(metricRegistry.timer(name(getClass(), indexName, "add")).getCount(), is(1L));
    }

    @Test
    void timesRemove() {
        decorator.remove(item1.id(), item1.version());
        verify(approximativeIndex).remove(item1.id(), item1.version());
        assertThat(metricRegistry.timer(name(getClass(), indexName, "remove")).getCount(), is(1L));
    }

    @Test
    void returnsSize() {
        int size = 10;
        given(approximativeIndex.size()).willReturn(size);
        assertThat(decorator.size(), is(size));
    }

    @Test
    void timesGet() {
        Optional<TestItem> getResult = Optional.of(this.item1);
        given(approximativeIndex.get(this.item1.id())).willReturn(getResult);
        assertThat(decorator.get(this.item1.id()), is(getResult));
        assertThat(metricRegistry.timer(name(getClass(), indexName, "get")).getCount(), is(1L));
    }

    @Test
    void timesContains() {
        given(approximativeIndex.contains(this.item1.id())).willReturn(true);
        assertThat(decorator.contains(this.item1.id()), is(true));
        assertThat(metricRegistry.timer(name(getClass(), indexName, "contains")).getCount(), is(1L));
    }

    @Test
    void returnsItems() {
        List<TestItem> items = Collections.singletonList(item1);
        given(approximativeIndex.items()).willReturn(items);
        assertThat(decorator.items(), is(items));
    }

    @Test
    void timesFindNearest() {
        List<SearchResult<TestItem, Float>> searchResults = Collections.singletonList(new SearchResult<>(item1, 0.1f, Comparator.naturalOrder()));

        given(approximativeIndex.findNearest(item1.vector(), k)).willReturn(searchResults);
        assertThat(decorator.findNearest(item1.vector(), k), is(searchResults));
        assertThat(metricRegistry.timer(name(getClass(), indexName, "findNearest")).getCount(), is(1L));
    }

    @Test
    void measuresFindNearestAccuracy() {
        List<SearchResult<TestItem, Float>> approximateResults = Collections.singletonList(
                new SearchResult<>(item1, 0.1f, Comparator.naturalOrder())
        );

        List<SearchResult<TestItem, Float>> groundTruthResults = Arrays.asList(
            new SearchResult<>(item1, 0.1f, Comparator.naturalOrder()),
            new SearchResult<>(item2, 0.1f, Comparator.naturalOrder())
        );

        given(approximativeIndex.findNearest(item1.vector(), k)).willReturn(approximateResults);
        given(groundTruthIndex.findNearest(item1.vector(), k)).willReturn(groundTruthResults);

        assertThat(decorator.findNearest(item1.vector(), k), is(approximateResults));
        await().untilAsserted(() -> assertThat(metricRegistry.histogram(name(getClass(), indexName, "accuracy")).getSnapshot().getMax(), is(50L)));
    }

    @Test
    void timesSave() throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        decorator.save(baos);
        assertThat(metricRegistry.timer(name(getClass(), indexName, "save")).getCount(), is(1L));
        verify(approximativeIndex).save(baos);
    }

    @Test
    void returnsApproximativeIndex() {
        assertThat(decorator.getApproximativeIndex(), is(approximativeIndex));
    }

    @Test
    void returnsGroundTruthIndex() {
        assertThat(decorator.getGroundTruthIndex(), is(groundTruthIndex));
    }
}

package org.github.jelmerk.hnsw;

import java.util.Comparator;
import java.util.List;

public class BinaryHeap<T extends Comparable<T>> {

    private final List<T> buffer;
    private final Comparator<T> comparator;

    /**
     * Initializes a new instance of the {@link BinaryHeap} class.
     *
     * @param buffer The buffer to store heap items.
     */
    BinaryHeap(List<T> buffer) {
        this(buffer, Comparator.naturalOrder());
    }

    /**
     * Initializes a new instance of the {@link BinaryHeap} class.
     *
     * @param buffer The buffer to store heap items.
     * @param comparer The comparer which defines order of items.
     */
    BinaryHeap(List<T> buffer, Comparator<T> comparer)  {
        if (buffer == null) {
            throw new IllegalArgumentException("buffer cannot be null");
        }

        this.buffer = buffer;
        this.comparator = comparer;
        for (int i = 1; i < this.buffer.size(); i++) {
            this.siftUp(i);
        }
    }

    /**
     * Gets the buffer of the heap.
     */
    public List<T> getBuffer() {
        return buffer;
    }

    /**
     * Gets the heap comparator.
     */
    public Comparator<T> getComparator() {
        return comparator;
    }

    /**
     * Pushes item to the heap.
     *
     * @param item The item to push.
     */
    void push(T item) {
        this.buffer.add(item);
        this.siftUp(this.buffer.size() - 1);
    }

    /**
     * Pops the item from the heap.
     *
     * @return The popped item.
     */
    T pop() {
        if (this.buffer.size() > 0) {
            T result = this.buffer.get(0);

            this.buffer.set(0, this.buffer.get(buffer.size() - 1));
            this.buffer.remove(this.buffer.size() - 1);
            this.siftDown(0);

            return result;
        }

        throw new IllegalStateException("Heap is empty");
    }

    /**
     * Restores the heap property starting from i'th position down to the bottom given that the downstream items
     * fulfill the rule.
     *
     * @param i The position of item where heap property is violated.
     */
    private void siftDown(int i) {
        while (i < this.buffer.size()) {
            int l = (i << 1) + 1;
            int r = l + 1;
            if (l >= this.buffer.size()) {
                break;
            }

            int m = r < this.buffer.size() && this.comparator.compare(this.buffer.get(l), this.buffer.get(r)) < 0 ? r : l;
            if (this.comparator.compare(this.buffer.get(m), this.buffer.get(i)) <= 0) {
                break;
            }

            this.swap(i, m);
            i = m;
        }
    }

    /**
     * Restores the heap property starting from i'th position up to the head given that the upstream items fulfill
     * the rule.
     *
     * @param i The position of item where heap property is violated.
     */
    private void siftUp(int i) {
        while (i > 0) {
            int p = (i - 1) >> 1;
            if (this.comparator.compare(this.buffer.get(i), this.buffer.get(p)) <= 0) {
                break;
            }

            this.swap(i, p);
            i = p;
        }
    }

    /**
     * Swaps items with the specified indices.
     * @param i The first index.
     * @param j The second index.
     */
    private void swap(int i, int j) {
        T temp = this.buffer.get(i);
        this.buffer.set(i, buffer.get(j));
        this.buffer.set(j, temp);
    }
}

package org.github.jelmerk.hnsw;

import java.util.Comparator;
import java.util.List;

public class BinaryHeap<T> {

    private final List<T> buffer;
    private final Comparator<T> comparator; // TODO rename to comparator

    /// <summary>
    /// Initializes a new instance of the <see cref="BinaryHeap{T}"/> class.
    /// </summary>
    /// <param name="buffer">The buffer to store heap items.</param>
    /// <param name="comparer">The comparer which defines order of items.</param>
    public BinaryHeap(List<T> buffer, Comparator<T> comparer)  {
        if (buffer == null) {
            throw new IllegalArgumentException("buffer cannot be null");
        }

        this.buffer = buffer;
        this.comparator = comparer;
        for (int i = 1; i < this.buffer.size(); i++) {
            this.siftUp(i);
        }
    }

    /// <summary>
    /// Gets the heap comparer.
    /// </summary>
    public Comparator<T> getComparator() {
        return comparator;
    }

    /// <summary>
    /// Gets the buffer of the heap.
    /// </summary>
    public List<T> getBuffer() {
        return buffer;
    }

    /// <summary>
    /// Pushes item to the heap.
    /// </summary>
    /// <param name="item">The item to push.</param>
    public void push(T item) {
        this.buffer.add(item);
        this.siftUp(this.buffer.size() - 1);
    }

    /// <summary>
    /// Pops the item from the heap.
    /// </summary>
    /// <returns>The popped item.</returns>
    public T pop() {
        if (!this.buffer.isEmpty()) {
            T result = this.buffer.get(0);

            this.buffer.set(0, this.buffer.get(buffer.size() - 1));
            this.buffer.remove(buffer.size() -1);

            this.siftDown(0);

            return result;
        }

        throw new IllegalStateException("Heap is empty");
    }

    /// <summary>
    /// Restores the heap property starting from i'th position down to the bottom
    /// given that the downstream items fulfill the rule.
    /// </summary>
    /// <param name="i">The position of item where heap property is violated.</param>
    private void siftDown(int i) {
        while (i < this.buffer.size())  {
            int l = (2 * i) + 1;
            int r = l + 1;

            if (l >= this.buffer.size()) {
                break;
            }

            int m = r < this.buffer.size() && this.comparator.compare(this.buffer.get(l), this.buffer.get(r)) < 0 ? r : l;
            if (this.comparator.compare(this.buffer.get(m), this.buffer.get(i)) <= 0)  {
                break;
            }

            this.swap(i, m);
            i = m;
        }
    }

    /// <summary>
    /// Restores the heap property starting from i'th position up to the head
    /// given that the upstream items fulfill the rule.
    /// </summary>
    /// <param name="i">The position of item where heap property is violated.</param>
    private void siftUp(int i) {
        while (i > 0) {
            int p = (i - 1) / 2;
            if (this.comparator.compare(this.buffer.get(i), this.buffer.get(p)) <= 0) {
                break;
            }

            this.swap(i, p);
            i = p;
        }
    }

    /// <summary>
    /// Swaps items with the specified indicies.
    /// </summary>
    /// <param name="i">The first index.</param>
    /// <param name="j">The second index.</param>
    private void swap(int i, int j) {
        T temp = this.buffer.get(i);
        this.buffer.set(i, buffer.get(j));
        this.buffer.set(j, temp);
    }

}

package org.github.jelmerk.hnsw;

import java.io.Serializable;

/**
 * Parameters of the algorithm.
 */

// TODO lets see if we can make this class immutable i dont think it makes sense to have a getter for parameters on algorithm and have it be mutable
public class Parameters implements Serializable {

    private int maxItemCount;
    private int m;
    private double levelLambda;
    private NeighbourSelectionHeuristic neighbourHeuristic;
    private int constructionPruning;
    private boolean expandBestSelection;
    private boolean keepPrunedConnections;

    /**
     * Initializes a new instance of the {@link Parameters} class.
     */
    public Parameters() {
        this.m = 10;
        this.levelLambda = 1 / Math.log(this.m);
        this.neighbourHeuristic = NeighbourSelectionHeuristic.SELECT_SIMPLE;
        this.constructionPruning = 200;
        this.expandBestSelection = false;
        this.keepPrunedConnections = true;
        this.maxItemCount = -1;
    }

    /**
     * Gets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
     *
     * The maximum number of neighbors for the zero layer is 2 * M.
     * The maximum number of neighbors for higher layers is M.
     */
    public int getM() {
        return m;
    }

    /**
     * Sets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
     *
     * The maximum number of neighbors for the zero layer is 2 * M.
     * The maximum number of neighbors for higher layers is M.
     */
    public void setM(int m) {
        this.m = m;
    }

    /**
     * Gets the max level decay parameter.
     *
     * @see <a href="https://en.wikipedia.org/wiki/Exponential_distribution">exponential distribution on wikipedia</a>
     * @see "'mL' parameter in the HNSW article."
     */
    public double getLevelLambda() {
        return levelLambda;
    }

    /**
     * Sets the max level decay parameter.
     *
     * @see <a href="https://en.wikipedia.org/wiki/Exponential_distribution">exponential distribution on wikipedia</a>
     * @see "'mL' parameter in the HNSW article."
     */
    public void setLevelLambda(double levelLambda) {
        this.levelLambda = levelLambda;
    }

    /**
     * Gets parameter which specifies the type of heuristic to use for best neighbours selection.
     */
    public NeighbourSelectionHeuristic getNeighbourHeuristic() {
        return neighbourHeuristic;
    }

    /**
     * Sets parameter which specifies the type of heuristic to use for best neighbours selection.
     */
    public void setNeighbourHeuristic(NeighbourSelectionHeuristic neighbourHeuristic) {
        this.neighbourHeuristic = neighbourHeuristic;
    }

    /**
     * Gets the number of candidates to consider as neighbours for a given node at the graph construction phase.
     *
     * @see "'efConstruction' parameter in the article."
     */
    public int getConstructionPruning() {
        return constructionPruning;
    }

    /**
     * Sets the number of candidates to consider as neighbours for a given node at the graph construction phase.
     *
     * @see "'efConstruction' parameter in the article."
     */
    public void setConstructionPruning(int constructionPruning) {
        this.constructionPruning = constructionPruning;
    }

    /**
     * Gets a value indicating whether to expand candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
     *
     * @see "'extendCandidates' parameter in the article."
     */
    public boolean isExpandBestSelection() {
        return expandBestSelection;
    }

    /**
     * Sets a value indicating whether to expand candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
     *
     * @see "'extendCandidates' parameter in the article."
     */
    public void setExpandBestSelection(boolean expandBestSelection) {
        this.expandBestSelection = expandBestSelection;
    }

    /**
     * Gets a value indicating whether to keep pruned candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
     *
     * @see "'keepPrunedConnections' parameter in the article."
     */
    public boolean isKeepPrunedConnections() {
        return keepPrunedConnections;
    }


    public int getMaxItemCount() {
        return maxItemCount;
    }

    public void setMaxItemCount(int maxItemCount) {
        this.maxItemCount = maxItemCount;
    }

    /**
     * Sets a value indicating whether to keep pruned candidates if {@link NeighbourSelectionHeuristic#SELECT_HEURISTIC} is used.
     *
     * @see "'keepPrunedConnections' parameter in the article."
     */
    public void setKeepPrunedConnections(boolean keepPrunedConnections) {
        this.keepPrunedConnections = keepPrunedConnections;
    }

}
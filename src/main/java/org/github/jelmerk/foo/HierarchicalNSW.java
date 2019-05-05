package org.github.jelmerk.foo;

import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

public class HierarchicalNSW<ID, ITEM extends Item<ID>> implements AlgorithmInterface<ID, ITEM> {

    private int m;
    private int maxM;
    private int maxM0;
    private int efConstruction;

    private double mult;
    private double revSize;
//    private int maxLevel;

//    VisitedListPool *visited_list_pool_;

//    private int enterpointNode;

    private Node enterpointNode;


    private DistanceFunction fstdistfunc;

    private final List<Item<ID>> items;
    private List<Node> nodes;
    private Map<ID, ITEM> idLookup;

    private Random levelGenerator;

    private int ef;
    private ReentrantLock global;

    public HierarchicalNSW(DistanceFunction fstdistfunc, int m, int efConstruction, int randomSeed) {

        this.fstdistfunc = fstdistfunc;

        this.m = m;
        this.maxM = m;
        this.maxM0 = m * 2;

        this.efConstruction = Math.max(efConstruction, m);

        this.ef = 10;

        this.levelGenerator = new Random(randomSeed);


        //initializations for special treatment of the first node
        this.enterpointNode = null;
//        this.maxLevel = -1;


        this.mult = 1 / Math.log(1d * m);
        this.revSize = 1d / this.mult;


        this.global = new ReentrantLock();

        this.items = Collections.synchronizedList(new ArrayList<>());
        this.nodes = Collections.synchronizedList(new ArrayList<>());
        this.idLookup = new ConcurrentHashMap<>();
    }


    @Override
    public ITEM getById(ID id) {
        return idLookup.get(id);
    }

    @Override
    public void addPoint(ITEM item) {
        addPoint(item, -1);
    }


    private int addPoint(ITEM item, int level) {

        int curC;


        int curlevel = getRandomLevel(mult);

        // TODO JK move this to constructor of node?
        List<List<Integer>> connections = new ArrayList<>(curlevel + 1);
        for (int layer = 0; layer <= curlevel; layer++) {

            int layerM = layer == 0 ? 2 * m : m;
            connections.add(new ArrayList<>(layerM));
        }

        synchronized (items) {
            curC = items.size();
            items.add(item);
            idLookup.put(item.getId(), item); // expected unique, if not will overwrite

            nodes.add(new Node(curC, connections));
        }

        synchronized (nodes.get(curC)) { // TODO i guess this could just immediately sync on the new node without lookup but intellij complains

            if (level > 0) {
                curlevel = level;  // TODO HOW does this even work, wouldnt it always become -1
            }

            global.lock();

            int maxLevelCopy = enterpointNode.maxLayer();

            if (curlevel <= maxLevelCopy) {
                global.unlock();
            }

            try {

                Node currObj = enterpointNode;

                if (currObj != null) {

                    if (curlevel < maxLevelCopy) {

                        float curDist = fstdistfunc.distance(item.getVector(), items.get(currObj.id).getVector());

                        for (int activeLevel = maxLevelCopy; activeLevel > curlevel; activeLevel--) {

                            boolean changed = true;

                            while (changed) {
                                changed = false;

                                synchronized (nodes.get(currObj.id)) { // TODO i guess this could just immediately sync on currObj without lookup but intellij complains

                                    for (Integer cand : currObj.connections.get(activeLevel - 1)) {
                                        float d = fstdistfunc.distance(item.getVector(), items.get(cand).getVector()); // TODO

                                        if (d < curDist) {
                                            curDist = d;
                                            currObj = nodes.get(cand);
                                            changed = true;
                                        }
                                    }

                                }

                            }
                        }

                        for (int activeLevel = Math.min(curlevel, maxLevelCopy); level >= 0; level--) {
                            if (activeLevel > maxLevelCopy || activeLevel < 0) {
                                throw new IllegalStateException("Level error");
                            }

                            PriorityQueue<Pair<Float, Integer>> topCandidates = searchBaseLayer(currObj, item, level);

                            mutuallyConnectNewElement(item, curC, topCandidates, level);
                        }

                    }

                } else {
                    // Do nothing for the first element
                    this.enterpointNode = nodes.get(curC); // TODO jk we already have the node, since we create it here no real need for lookup
                }

                //Releasing lock for the maximum level
                if (curlevel > maxLevelCopy) {
                    this.enterpointNode = nodes.get(curC); // TODO jk i guess this is the same as above
                }
                return curC;


            } finally {
                if (global.isHeldByCurrentThread()) {
                    global.unlock();
                }
            }
        }

    }




    PriorityQueue<Pair<Float, Integer>> searchBaseLayer(Node enterPoint, float[] datapoint, int layer) {



    }




/*

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint enterpoint_id, void *data_point, int layer) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(enterpoint_id), dist_func_param_);

            top_candidates.emplace(dist, enterpoint_id);
            candidateSet.emplace(-dist, enterpoint_id);
            visited_array[enterpoint_id] = visited_array_tag;
            dist_t lowerBound = dist;

            while (!candidateSet.empty()) {

                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();

                if ((-curr_el_pair.first) > lowerBound) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                int *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                if (layer == 0)
                    data = (int *) (data_level0_memory_ + curNodeNum * size_data_per_element_ + offsetLevel0_);
                else
                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                int size = *data;
                tableint *datal = (tableint *) (data + 1);
        #ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
        #endif

                for (int j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
        #ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
        #endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.top().first > dist1 || top_candidates.size() < ef_construction_) {
                        candidateSet.emplace(-dist1, candidate_id);
        #ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
        #endif
                        top_candidates.emplace(dist1, candidate_id);
                        if (top_candidates.size() > ef_construction_) {
                            top_candidates.pop();
                        }
                        lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }


 */






    private void getNeighborsByHeuristic2(PriorityQueue<Pair<Float, Integer>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<Pair<Float, Integer>> queueClosest = new PriorityQueue<>();
        List<Pair<Float, Integer>> returnList = new ArrayList<>();

        while(!topCandidates.isEmpty()) {
            Pair<Float, Integer> element = topCandidates.remove();
            queueClosest.add(Pair.of(-element.getFirst(), element.getSecond()));
        }

        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            Pair<Float, Integer> currentPair = queueClosest.remove();

            float distToQuery = -currentPair.getFirst();

            boolean good = true;
            for (Pair<Float, Integer> secondPair : returnList) {

                float curdist = fstdistfunc.distance(
                        getDataByInternalId(secondPair.getSecond()),
                        getDataByInternalId(currentPair.getSecond())
                );

                if (curdist < distToQuery) {
                    good = false;
                    break;
                }

            }
            if (good) {
                returnList.add(currentPair);
            }
        }

        for (Pair<Float, Integer> currentPair : returnList) {
            topCandidates.add(new Pair<>(-currentPair.getFirst(), currentPair.getSecond()));
        }

    }

//    void getNeighborsByHeuristic2(
//            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
//                const size_t M) {
//        if (top_candidates.size() < M) {
//            return;
//        }
//        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
//        std::vector<std::pair<dist_t, tableint>> return_list;
//        while (top_candidates.size() > 0) {
//            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
//            top_candidates.pop();
//        }
//
//        while (queue_closest.size()) {
//            if (return_list.size() >= M)
//                break;
//            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
//            dist_t dist_to_query = -curent_pair.first;
//            queue_closest.pop();
//            bool good = true;
//            for (std::pair<dist_t, tableint> second_pair : return_list) {
//                dist_t curdist =
//                        fstdistfunc_(getDataByInternalId(second_pair.second),
//                                getDataByInternalId(curent_pair.second),
//                                dist_func_param_);;
//                if (curdist < dist_to_query) {
//                    good = false;
//                    break;
//                }
//            }
//            if (good) {
//                return_list.push_back(curent_pair);
//            }
//
//
//        }
//
//        for (std::pair<dist_t, tableint> curent_pair : return_list) {
//
//            top_candidates.emplace(-curent_pair.first, curent_pair.second);
//        }
//    }



    private void mutuallyConnectNewElement(float[] dataPoint,
                                           int curC,
                                           PriorityQueue<Pair<Float, Integer>> topCandidates,
                                           int level) {

        int mCurMax = level != 0 ? maxM : maxM0;

        getNeighborsByHeuristic2(topCandidates, this.m);
        if (topCandidates.size() > m) {
            throw new IllegalStateException("Should be not be more than m candidates returned by the heuristic");
        }

        List<Integer> selectedNeighbors = new ArrayList<>(m);

        while(!topCandidates.isEmpty()) {
            selectedNeighbors.add(topCandidates.remove().getSecond());
        }

        // TODO: JK no clue what to do with this

//        {
//            linklistsizeint *ll_cur;
//            if (level == 0)
//                ll_cur = get_linklist0(cur_c);
//            else
//                ll_cur = get_linklist(cur_c, level);
//
//            if (*ll_cur) {
//            throw std::runtime_error("The newly inserted element should have blank link list");
//        }


        for (int idx = 0; idx < selectedNeighbors.size(); idx++) {

            if (level > this.elementLevels.get(selectedNeighbors.get(idx))) {
                throw new IllegalStateException("Trying to make a link on a non-existent level");
            }


        }





//adaro: sorry for all the stupid questions but what does this do : tableint *data = (tableint *) (ll_cur + 1);  is that declaring an array of tableints with size  ll_cur + 1 ?
//TinoDidriksen: No, it's pointing to the single tableint at offset ll_cur+1


//adaro: So in my case if [1,2,3] was the underlying array and data is defined as tableint *data and its value is 2 , would  data[1] = 4 then change 3 to 4 in the underlying array ?
//TinoDidriksen: Yes




//        *ll_cur = selectedNeighbors.size();
//        tableint *data = (tableint *) (ll_cur + 1);

//            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
//                if (data[idx])
//                    throw std::runtime_error("Possible memory corruption");
//                if (level > element_levels_[selectedNeighbors[idx]])
//                    throw std::runtime_error("Trying to make a link on a non-existent level");
//
//                data[idx] = selectedNeighbors[idx];
//
//            }


    }


//    void mutuallyConnectNewElement(void *data_point, tableint cur_c,
//                                   std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates,
//                                   int level) {
//
//        size_t Mcurmax = level ? maxM_ : maxM0_;
//        getNeighborsByHeuristic2(top_candidates, M_);
//        if (top_candidates.size() > M_)
//            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");
//
//        std::vector<tableint> selectedNeighbors;
//        selectedNeighbors.reserve(M_);
//        while (top_candidates.size() > 0) {
//            selectedNeighbors.push_back(top_candidates.top().second);
//            top_candidates.pop();
//        }
//        {
//            linklistsizeint *ll_cur;
//            if (level == 0)
//                ll_cur = get_linklist0(cur_c);
//            else
//                ll_cur = get_linklist(cur_c, level);
//
//            if (*ll_cur) {
//            throw std::runtime_error("The newly inserted element should have blank link list");
//        }
//                *ll_cur = selectedNeighbors.size();
//            tableint *data = (tableint *) (ll_cur + 1);
//
//
//            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
//                if (data[idx])
//                    throw std::runtime_error("Possible memory corruption");
//                if (level > element_levels_[selectedNeighbors[idx]])
//                    throw std::runtime_error("Trying to make a link on a non-existent level");
//
//                data[idx] = selectedNeighbors[idx];
//
//            }
//        }
//        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
//
//            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);
//
//
//            linklistsizeint *ll_other;
//            if (level == 0)
//                ll_other = get_linklist0(selectedNeighbors[idx]);
//            else
//                ll_other = get_linklist(selectedNeighbors[idx], level);
//            size_t sz_link_list_other = *ll_other;
//
//
//            if (sz_link_list_other > Mcurmax)
//                throw std::runtime_error("Bad value of sz_link_list_other");
//            if (selectedNeighbors[idx] == cur_c)
//                throw std::runtime_error("Trying to connect an element to itself");
//            if (level > element_levels_[selectedNeighbors[idx]])
//                throw std::runtime_error("Trying to make a link on a non-existent level");
//
//            tableint *data = (tableint *) (ll_other + 1);
//            if (sz_link_list_other < Mcurmax) {
//                data[sz_link_list_other] = cur_c;
//                    *ll_other = sz_link_list_other + 1;
//            } else {
//                // finding the "weakest" element to replace it with the new one
//                dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
//                        dist_func_param_);
//                // Heuristic:
//                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
//                candidates.emplace(d_max, cur_c);
//
//                for (size_t j = 0; j < sz_link_list_other; j++) {
//                    candidates.emplace(
//                            fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
//                                    dist_func_param_), data[j]);
//                }
//
//                getNeighborsByHeuristic2(candidates, Mcurmax);
//
//                int indx = 0;
//                while (candidates.size() > 0) {
//                    data[indx] = candidates.top().second;
//                    candidates.pop();
//                    indx++;
//                }
//                    *ll_other = indx;
//                // Nearest K:
//                    /*int indx = -1;
//                    for (int j = 0; j < sz_link_list_other; j++) {
//                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
//                        if (d > d_max) {
//                            indx = j;
//                            d_max = d;
//                        }
//                    }
//                    if (indx >= 0) {
//                        data[indx] = cur_c;
//                    } */
//            }
//
//        }
//    }





    @Override
    public PriorityQueue<SearchResult<ITEM>> searchKnn(float[] vector, int k) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void saveIndex(OutputStream out) throws IOException {

    }

    private int getRandomLevel(double reverseSize) {
        double r = -Math.log(levelGenerator.nextDouble()) * reverseSize;
        return (int) r;
    }


    static class Node  {

        private int id;

        private List<List<Integer>> connections;

        public Node(int id, List<List<Integer>> connections) {
            this.id = id;
            this.connections = connections;
        }

        public int maxLayer() {
            return this.connections.size() - 1;
        }
    }

}

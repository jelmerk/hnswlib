package org.github.jelmerk.foo;

import java.io.IOException;
import java.io.OutputStream;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

public class HierarchicalNSW<TItem, TDistance extends Comparable<TDistance>>
        implements AlgorithmInterface<TItem, TDistance> {


    /*

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;
        tableint enterpoint_node_;



        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;


        size_t data_size_;
        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;


     */


    private final SpaceInterface spaceInterface;

    private final int maxElements;

    private int curElementCount;

    private final int m;
    private final int efConstruction;


    private final int dataSize;
    private final DistanceFunction<TItem, TDistance> distanceFunction;
    private final int maxM0;

    private final Random levelGenerator;
    private final int ef;
    private final int maxM;
    private final int offsetLevel0;

    private int enterpointNode;
    private int maxLevel;



    private double mult;
    private double revSize;



//    private final ReentrantLock currentElementCountGuard;
    private final ReentrantLock global;

    private final Object currentElementCountGuard = new Object();


    private List<Integer> elementLevels; // TODO JK can we use a primitive array here and use maxElements as the size ?



    private List<TItem> items; // JK added thia myself

    public HierarchicalNSW(SpaceInterface spaceInterface, int maxElements, int m, int efConstruction, int randomSeed) {
        this.spaceInterface = spaceInterface;
        this.maxElements = maxElements;



        this.dataSize = spaceInterface.getDataSize();
        this.distanceFunction = spaceInterface.getDistanceFunction();
//        dist_func_param_ = s->get_dist_func_param();  TODO JK not sure what this is
        this.m = m;
        this.maxM = m;
        this.maxM0 = m * 2;

        this.efConstruction = Math.max(efConstruction, m);

        this.ef = 10;

        this.levelGenerator = new Random(randomSeed);

        this.offsetLevel0 = 0;

        //initializations for special treatment of the first node
        this.enterpointNode = -1;
        this.maxLevel = -1;


        this.mult = 1 / Math.log(1d * m);
        this.revSize = 1d / this.mult;


        this.curElementCount = 0;
        this.global = new ReentrantLock();
        /*

            level_generator_.seed(random_seed);

            size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
            offsetData_ = size_links_level0_;
            label_offset_ = size_links_level0_ + data_size_;
            offsetLevel0_ = 0;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);



            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

         */


        this.elementLevels = new ArrayList<>();

        this.items = Collections.synchronizedList(new ArrayList<>()); // synchronuzed i guess since we need to read and write from it


    }

    @Override
    public void addPoint(TItem item) {
        addPoint(item, -1);
    }

    private int addPoint(TItem item, int level) {

        int curC;

        synchronized (currentElementCountGuard) {

            if (curElementCount >= maxElements) {
                throw new IllegalArgumentException("The number of elements exceeds the specified limit.");
            }

            items.set(curElementCount, item);

            curC = curElementCount;
//        label_lookup_[label] = cur_c;  // expected unique, if not will overwrite TODO: JK not sure what to do with this
            curElementCount++;

        }



        int curlevel = getRandomLevel(mult);

        if (level > 0) {
            curlevel = level;
        }

        elementLevels.set(curC, curlevel);


        global.lock(); // TODO make sure we unlock this

        int maxLevelCopy = this.maxLevel;

        if (curlevel <= maxLevelCopy) {
            global.unlock();
        }

        int currObj = enterpointNode;




        /*
            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the data and label
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        */


        // if (c) is the same as if (c != 0). And if (!c) is the same as if (c == 0).

        if (curlevel != 0) {
//            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
//            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if (currObj != -1) {

            if (curlevel < maxLevelCopy) {

                // TODO JK , the original code passes in spaceInterface.get_dist_func_param as 3rd argument to the distance function
                // TODO JK i guess we need to find the item for the id here or something in getDataByInternalId

                TDistance curDist = distanceFunction.distance(item, items.get(currObj));

                for (int activeLevel = maxLevelCopy; activeLevel > curlevel; activeLevel--) {

                    boolean changed = true;

                    while(changed) {
                        changed = false;

                        synchronized (items.get(currObj)) { // TODO jk : ehm i guess just getting the item from an unsynchronized list is not thread safe.. though if its an array what could go wrong




                            for (int i = 0; i < size; i++) {

                                int cand = ???
                                if (cand < 0 || cand > maxElements) {
                                    throw new IllegalStateException("cand error");
                                }

                                TDistance d = distanceFunction.distance(item, items.get(cand)); // TODO

                                if (d.compareTo(curDist) < 0) {
                                    curDist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }


                        }

                    }
                }

                for (int level = Math.min(curlevel, maxLevelCopy); level >= 0; level--) {
                    if (level > maxLevelCopy || level < 0) {
                        throw new IllegalStateException("Level error");
                    }

                    PriorityQueue<Pair<TDistance, Integer>> topCandidates = searchBaseLayer(currObj, item, level);

                    mutuallyConnectNewElement(item, curC, topCandidates, level);
                }

            }

        } else {
            // Do nothing for the first element
            this.enterpointNode = 0;
            this.maxLevel = curlevel;
        }

        //Releasing lock for the maximum level
        if (curlevel > maxLevelCopy) {
            this.enterpointNode = curC;
            this.maxLevel = curlevel;
        }
        return curC;


        /*

            if ((signed)currObj != -1) {


                if (curlevel < maxlevelcopy) {

                    dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                    for (int level = maxlevelcopy; level > curlevel; level--) {


                        bool changed = true;
                        while (changed) {
                            changed = false;
                            int *data;
                            std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                            data = (int *) (linkLists_[currObj] + (level - 1) * size_links_per_element_);
                            int size = *data;
                            tableint *datal = (tableint *) (data + 1);
                            for (int i = 0; i < size; i++) {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                if (d < curdist) {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)
                        throw std::runtime_error("Level error");

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                            currObj, data_point, level);
                    mutuallyConnectNewElement(data_point, cur_c, top_candidates, level);
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
         */

    }



    void getNeighborsByHeuristic2(PriorityQueue<Pair<TDistance, Integer>> topCandidates, int m) {

        if (topCandidates.size() < m) {
            return;
        }

        PriorityQueue<Pair<TDistance, Integer>> queueClosest = new PriorityQueue<>();
        List<Pair<TDistance, Integer>> returnList = new ArrayList<>();


        // TODO: JK i guess this just reverses the queue ?
        while(!topCandidates.isEmpty()) {
            Pair<TDistance, Integer> element = topCandidates.remove();
            queueClosest.add(element);
        }

        while(!queueClosest.isEmpty()) {
            if (returnList.size() >= m) {
                break;
            }

            Pair<TDistance, Integer> currentPair = queueClosest.remove();

            TDistance distanceToQuery = -currentPair.getFirst(); // TODO this is made negative in the original code i think its written as -curent_pair.first

            boolean good = true;
            for (Pair<TDistance, Integer> secondPair : returnList) {

                TItem secondItem = this.items.get(secondPair.getSecond());
                TItem firstItem = this.items.get(currentPair.getSecond());

                TDistance curdist = distanceFunction.distance(secondItem, firstItem);

                if (curdist.compareTo(distanceToQuery) < 0) {
                    good = false;
                    break;
                }

            }
            if (good) {
                returnList.add(currentPair);
            }
        }

        for (Pair<TDistance, Integer> currentPair : returnList) {
            topCandidates.add(new Pair<>(-currentPair.getFirst(), currentPair.getSecond())); // TODO this is made negative in the original code top_candidates.emplace(-curent_pair.first, curent_pair.second);
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


    // TODO jk :looks like this is all done with direct memory access..
    // TODO data_level0_memory_ is defined as data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
    // TODO i guess i could do the same to safe on memory.. and use off heap stuff maybe


//    linklistsizeint *get_linklist0(tableint internal_id) {
//        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
//    };
//
//    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) {
//        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
//    };
//
//    linklistsizeint *get_linklist(tableint internal_id, int level) {
//        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
//    };

    private void mutuallyConnectNewElement(TItem dataPoint,
                                           int curC,
                                           PriorityQueue<Pair<TDistance, Integer>> topCandidates,
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
    public PriorityQueue<TDistance> searchKnn(TItem tItem, int k) {
        return null;
    }

    @Override
    public void saveIndex(OutputStream out) throws IOException {

    }

    private int getRandomLevel(double reverseSize) {
        double r = -Math.log(levelGenerator.nextDouble()) * reverseSize;
        return (int) r;
    }


}

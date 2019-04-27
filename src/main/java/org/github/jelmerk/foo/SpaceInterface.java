package org.github.jelmerk.foo;

public interface SpaceInterface<TItem, TDistance extends Comparable<TDistance>> {

    int getDataSize();

    DistanceFunction<TItem, TDistance> getDistanceFunction();

//    Object getDistanceFunctionParam(); // TODO this seems to be some extra value you can pass to the distance function. not sure if we need this or we can just ommit this

}


/*

        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

        virtual ~SpaceInterface() {}
 */
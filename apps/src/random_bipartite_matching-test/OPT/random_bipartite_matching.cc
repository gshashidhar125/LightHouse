#include "random_bipartite_matching.h"

int32_t random_bipartite_matching(gm_graph& G, bool* G_isLeft, 
    node_t* G_Match)
{
    //Initializations
    gm_rt_initialize();
    G.freeze();

    int32_t count = 0 ;
    bool finished = false ;
    node_t* G_Suitor = gm_rt_allocate_node_t(G.num_nodes(),gm_rt_thread_id());

    count = 0 ;
    finished = false ;

    #pragma omp parallel for
    for (node_t t0 = 0; t0 < G.num_nodes(); t0 ++) 
    {
        G_Match[t0] = gm_graph::NIL_NODE ;
        G_Suitor[t0] = gm_graph::NIL_NODE ;
    }
    while ( !finished)
    {
        finished = false ;
        #pragma omp parallel
        {
            bool finished_prv = false ;

            finished_prv = true ;

            #pragma omp for nowait schedule(dynamic,128)
            for (node_t n = 0; n < G.num_nodes(); n ++) 
            {
                if (G_isLeft[n] && (G_Match[n] == gm_graph::NIL_NODE))
                {
                    for (edge_t t_idx = G.begin[n];t_idx < G.begin[n+1] ; t_idx ++) 
                    {
                        node_t t = G.node_idx [t_idx];
                        if (G_Match[t] == gm_graph::NIL_NODE)
                        {
                            G_Suitor[t] = n ;
                            finished_prv = finished_prv && false ;
                        }
                    }
                }
            }
            ATOMIC_AND(&finished, finished_prv);
        }

        #pragma omp parallel for
        for (node_t t2 = 0; t2 < G.num_nodes(); t2 ++) 
        {
            if ( !G_isLeft[t2] && (G_Match[t2] == gm_graph::NIL_NODE))
            {
                if (G_Suitor[t2] != gm_graph::NIL_NODE)
                {
                    node_t n3;

                    n3 = G_Suitor[t2] ;
                    G_Suitor[n3] = t2 ;
                    G_Suitor[t2] = gm_graph::NIL_NODE ;
                }
            }
        }
        #pragma omp parallel
        {
            int32_t count_prv = 0 ;

            count_prv = 0 ;

            #pragma omp for nowait
            for (node_t n4 = 0; n4 < G.num_nodes(); n4 ++) 
            {
                if (G_isLeft[n4] && (G_Match[n4] == gm_graph::NIL_NODE))
                {
                    if (G_Suitor[n4] != gm_graph::NIL_NODE)
                    {
                        node_t t5;

                        t5 = G_Suitor[n4] ;
                        G_Match[n4] = t5 ;
                        G_Match[t5] = n4 ;
                        count_prv = count_prv + 1 ;
                    }
                }
            }
            ATOMIC_ADD<int32_t>(&count, count_prv);
        }
    }
    gm_rt_cleanup();
    return count; 
}


#define GM_DEFINE_USER_MAIN 1
#if GM_DEFINE_USER_MAIN

// random_bipartite_matching -? : for how to run generated main program
int main(int argc, char** argv)
{
    gm_default_usermain Main;

    Main.declare_property("G_isLeft", GMTYPE_BOOL, true, false, GM_NODEPROP);
    Main.declare_property("G_Match", GMTYPE_NODE, true, true, GM_NODEPROP);
    Main.declare_return(GMTYPE_INT);

    if (!Main.process_arguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    if (!Main.do_preprocess()) {
        return EXIT_FAILURE;
    }

    Main.begin_usermain();
    Main.set_return_i(random_bipartite_matching(
        Main.get_graph(),
        (bool*) Main.get_property("G_isLeft"),
        (node_t*) Main.get_property("G_Match")
    )
);
Main.end_usermain();

if (!Main.do_postprocess()) {
    return EXIT_FAILURE;
}

return EXIT_SUCCESS;
}
#endif

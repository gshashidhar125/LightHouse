#include "v_cover.h"

int32_t v_cover(gm_graph& G, bool* G_selectEdge)
{
    //Initializations
    gm_rt_initialize();
    G.freeze();
    G.make_reverse_edges();

    int32_t remain = 0 ;
    int32_t __S4 = 0 ;
    int32_t* G_Deg = gm_rt_allocate_int(G.num_nodes(),gm_rt_thread_id());
    bool* G_Covered = gm_rt_allocate_bool(G.num_nodes(),gm_rt_thread_id());

    remain = (int32_t)(G.num_edges() * 2) ;
    __S4 = 0 ;

    #pragma omp parallel for
    for (node_t t0 = 0; t0 < G.num_nodes(); t0 ++) 
    {
        G_Deg[t0] = (G.begin[t0+1] - G.begin[t0]) + (G.r_begin[t0+1] - G.r_begin[t0]) ;
        G_Covered[t0] = false ;
    }

    #pragma omp parallel for
    for (edge_t t2 = 0; t2 < G.num_edges(); t2 ++) 
        G_selectEdge[t2] = false ;

    while (remain > 0)
    {
        int32_t max_val = 0 ;
        node_t from;
        node_t to;
        edge_t e;

        max_val = 0 ;
        #pragma omp parallel
        {
            edge_t e_prv;
            node_t to_prv;
            node_t from_prv;
            int32_t max_val_prv = 0 ;

            max_val_prv = max_val ;

            #pragma omp for nowait schedule(dynamic,128)
            for (node_t s = 0; s < G.num_nodes(); s ++) 
            {
                for (edge_t t_idx = G.begin[s];t_idx < G.begin[s+1] ; t_idx ++) 
                {
                    node_t t = G.node_idx [t_idx];
                    if ( !(G_Covered[s] && G_Covered[t]))
                    {
                        if (max_val_prv < G_Deg[s] + G_Deg[t])
                        {
                            max_val_prv = G_Deg[s] + G_Deg[t] ;
                            from_prv = s ;
                            to_prv = t ;
                            e_prv = t_idx ;
                        }
                    }
                }
            }
            { // argmin(argmax) - test and test-and-set
                int32_t max_val_new = max_val_prv;
                if (max_val<max_val_new) {
                    node_t from_arg = from_prv;
                    node_t to_arg = to_prv;
                    edge_t e_arg = e_prv;
                    gm_spinlock_acquire_for_ptr(&max_val);
                    if (max_val<max_val_new) {
                        max_val = max_val_new;
                        from = from_arg;
                        to = to_arg;
                        e = e_arg;
                    }
                    gm_spinlock_release_for_ptr(&max_val);
                }
            }
        }
        G_Covered[from] = true ;
        G_Covered[to] = true ;
        G_selectEdge[e] = true ;
        G_Deg[from] = 0 ;
        G_Deg[to] = 0 ;
        remain = remain - max_val ;
    }
    #pragma omp parallel
    {
        int32_t __S4_prv = 0 ;

        __S4_prv = 0 ;

        #pragma omp for nowait
        for (node_t t3 = 0; t3 < G.num_nodes(); t3 ++) 
            if (G_Covered[t3])
            {
                __S4_prv = __S4_prv + 1 ;
            }

        ATOMIC_ADD<int32_t>(&__S4, __S4_prv);
    }
    gm_rt_cleanup();
    return __S4; 
}


#define GM_DEFINE_USER_MAIN 1
#if GM_DEFINE_USER_MAIN

// v_cover -? : for how to run generated main program
int main(int argc, char** argv)
{
    gm_default_usermain Main;

    Main.declare_property("G_selectEdge", GMTYPE_BOOL, true, true, GM_EDGEPROP);
    Main.declare_return(GMTYPE_INT);

    if (!Main.process_arguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    if (!Main.do_preprocess()) {
        return EXIT_FAILURE;
    }

    Main.begin_usermain();
    Main.set_return_i(v_cover(
        Main.get_graph(),
        (bool*) Main.get_property("G_selectEdge")
    )
);
Main.end_usermain();

if (!Main.do_postprocess()) {
    return EXIT_FAILURE;
}

return EXIT_SUCCESS;
}
#endif

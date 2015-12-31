#include "hop_dist.h"

void hop_dist(gm_graph& G, int32_t* G_dist, 
    node_t& root)
{
    //Initializations
    gm_rt_initialize();
    G.freeze();

    bool fin = false ;
    bool* G_updated = gm_rt_allocate_bool(G.num_nodes(),gm_rt_thread_id());
    bool* G_updated_nxt = gm_rt_allocate_bool(G.num_nodes(),gm_rt_thread_id());
    int32_t* G_dist_nxt = gm_rt_allocate_int(G.num_nodes(),gm_rt_thread_id());

    fin = false ;

    #pragma omp parallel for
    for (node_t t0 = 0; t0 < G.num_nodes(); t0 ++) 
    {
        G_dist[t0] = (t0 == root)?0:INT_MAX ;
        G_updated[t0] = (t0 == root)?true:false ;
        G_dist_nxt[t0] = G_dist[t0] ;
        G_updated_nxt[t0] = G_updated[t0] ;
    }
    while ( !fin)
    {
        bool __E8 = false ;

        fin = true ;
        __E8 = false ;

        #pragma omp parallel for schedule(dynamic,128)
        for (node_t n = 0; n < G.num_nodes(); n ++) 
        {
            if (G_updated[n])
            {
                for (edge_t s_idx = G.begin[n];s_idx < G.begin[n+1] ; s_idx ++) 
                {
                    node_t s = G.node_idx [s_idx];
                    { // argmin(argmax) - test and test-and-set
                        int32_t G_dist_nxt_new = G_dist[n] + 1;
                        if (G_dist_nxt[s]>G_dist_nxt_new) {
                            bool G_updated_nxt_arg = true;
                            gm_spinlock_acquire_for_node(s);
                            if (G_dist_nxt[s]>G_dist_nxt_new) {
                                G_dist_nxt[s] = G_dist_nxt_new;
                                G_updated_nxt[s] = G_updated_nxt_arg;
                            }
                            gm_spinlock_release_for_node(s);
                        }
                    }
                }
            }
        }
        #pragma omp parallel
        {
            bool __E8_prv = false ;

            __E8_prv = false ;

            #pragma omp for nowait
            for (node_t t4 = 0; t4 < G.num_nodes(); t4 ++) 
            {
                G_dist[t4] = G_dist_nxt[t4] ;
                G_updated[t4] = G_updated_nxt[t4] ;
                G_updated_nxt[t4] = false ;
                __E8_prv = __E8_prv || G_updated[t4] ;
            }
            ATOMIC_OR(&__E8, __E8_prv);
        }
        fin =  !__E8 ;
    }

    gm_rt_cleanup();
}


#define GM_DEFINE_USER_MAIN 1
#if GM_DEFINE_USER_MAIN

// hop_dist -? : for how to run generated main program
int main(int argc, char** argv)
{
    gm_default_usermain Main;

    Main.declare_property("G_dist", GMTYPE_INT, true, true, GM_NODEPROP);
    Main.declare_scalar("root", GMTYPE_NODE, true, false);

    if (!Main.process_arguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    if (!Main.do_preprocess()) {
        return EXIT_FAILURE;
    }

    Main.begin_usermain();
    hop_dist(
        Main.get_graph(),
        (int32_t*) Main.get_property("G_dist"),
        *((node_t*) Main.get_scalar("root")));
    Main.end_usermain();

    if (!Main.do_postprocess()) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
#endif

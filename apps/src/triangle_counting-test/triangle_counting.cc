#include "triangle_counting.h"

int64_t triangle_counting(gm_graph& G)
{
    //Initializations
    gm_rt_initialize();
    G.freeze();
    G.do_semi_sort();

    int64_t T = 0 ;

    T = 0 ;
    #pragma omp parallel
    {
        int64_t T_prv = 0 ;

        T_prv = 0 ;

        #pragma omp for nowait schedule(dynamic,128)
        for (node_t v = 0; v < G.num_nodes(); v ++) 
            for (edge_t u_idx = G.begin[v];u_idx < G.begin[v+1] ; u_idx ++) 
            {
                node_t u = G.node_idx [u_idx];
                if (u > v)
                {
                    for (edge_t w_idx = G.begin[v];w_idx < G.begin[v+1] ; w_idx ++) 
                    {
                        node_t w = G.node_idx [w_idx];
                        if (w > u)
                        {
                            if (G.is_neighbor(w,u))
                            {
                                T_prv = T_prv + 1 ;
                            }
                        }
                    }
                }
            }

        ATOMIC_ADD<int64_t>(&T, T_prv);
    }
    return T; 
}


#define GM_DEFINE_USER_MAIN 1
#if GM_DEFINE_USER_MAIN

// triangle_counting -? : for how to run generated main program
int main(int argc, char** argv)
{
    gm_default_usermain Main;

    Main.declare_return(GMTYPE_LONG);

    if (!Main.process_arguments(argc, argv)) {
        return EXIT_FAILURE;
    }

    if (!Main.do_preprocess()) {
        return EXIT_FAILURE;
    }

    Main.begin_usermain();
    Main.set_return_l(triangle_counting(
        Main.get_graph()
    )
);
Main.end_usermain();

if (!Main.do_postprocess()) {
    return EXIT_FAILURE;
}

return EXIT_SUCCESS;
}
#endif

#include "gm_compile_step.h"

#ifndef GM_BACKEND_CUDA_GEN_STEPS_H
#define GM_BACKEND_CUDA_GEN_STEPS_H

//-------------------------------------------
// [Step 1]
// Add declaration of step here
//-------------------------------------------
/*GM_COMPILE_STEP(gm_cuda_gen_sanitize_name, "Sanitize identifier")
GM_COMPILE_STEP(gm_cuda_gen_regular, "Regularize code structure for code generation")
GM_COMPILE_STEP(gm_cuda_gen_prop_decl, "Check property declaration");
GM_COMPILE_STEP(gm_cuda_gen_mark_parallel, "Mark every parallel sentence")
GM_COMPILE_STEP(gm_cuda_gen_check_bfs, "Check BFS routines")
GM_COMPILE_STEP(gm_cuda_gen_misc_check, "Checking small assumptions")
GM_COMPILE_STEP(gm_cuda_gen_check_usermain, "Checking items for user main generation")
GM_COMPILE_STEP(gm_cuda_gen_check_rev_edge, "Checking reverse edges") */
GM_COMPILE_STEP(gm_cuda_gen_proc, "Creating CUDA source for each procedure")

//-------------------------------------------
// [Step 2]
// Implement step::process() in a seperate file
//-------------------------------------------

//------------------------------------------------------
// [Step 3]
// Add step in module initialization (gm_cuda_gen_main.cc)
//------------------------------------------------------

#endif


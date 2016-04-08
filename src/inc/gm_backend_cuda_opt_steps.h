#include "gm_compile_step.h"
#include "gm_backend_cuda.h"

#ifndef GM_BACKEND_CUDA_OPT_STEPS_H
#define GM_BACKEND_CUDA_OPT_STEPS_H

//-------------------------------------------
// [Step 1]
// Add delaration here
// declaration of optimization steps
//-------------------------------------------
/*GM_COMPILE_STEP(gm_cuda_opt_check_feasible, "Check compiler feasiblity")
GM_COMPILE_STEP(gm_cuda_opt_defer, "Handle deferred writes")
GM_COMPILE_STEP(gm_cuda_opt_sanitize_name, "Sanitize identifier")
GM_COMPILE_STEP(gm_cuda_opt_common_nbr, "Common Neigbhor Iteration")
GM_COMPILE_STEP(gm_cuda_opt_select_par, "Select parallel regions")
GM_COMPILE_STEP(gm_cuda_opt_save_bfs, "Finding BFS Children")
//GM_COMPILE_STEP(gm_cuda_opt_reduce_bound, "Optimize reductions with sequential bound ")
GM_COMPILE_STEP(gm_cuda_opt_reduce_scalar, "Privitize reduction to scalar")
GM_COMPILE_STEP(gm_cuda_opt_reduce_field, "Privitize reduction to field")
GM_COMPILE_STEP(gm_cuda_opt_temp_cleanup, "Clean-up routines for temporary properties")
GM_COMPILE_STEP(gm_cuda_opt_select_seq_implementation, "Select implementation for Node_Seq/Edge_Seq")
GM_COMPILE_STEP(gm_cuda_opt_select_map_implementation, "Select implementation for Map")
GM_COMPILE_STEP(gm_cuda_opt_entry_exit, "Add procedure enter and exit")
GM_COMPILE_STEP(gm_cuda_opt_debug, "A dummy routine for debug")
*/
GM_COMPILE_STEP(gm_cuda_opt_dependencyAnalysis, "Generate Dependency(Def/Use) Infomation")
GM_COMPILE_STEP(gm_cuda_opt_loopColapse, "Replace a nested forEach loop going over neighbours of all the nodes to a forEach loop over edges")
GM_COMPILE_STEP(gm_cuda_opt_removeAtomicsForBoolean, "Eliminate Boolean atomic reduction instructions")
//-------------------------------------------
// [Step 2]
//   Implement the definition in seperate files
//-------------------------------------------

//------------------------------------------------------
// [Step 3]
//   Include initialization in following steps
//------------------------------------------------------

#endif


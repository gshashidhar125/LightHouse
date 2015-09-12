#include <stdio.h>
#include "gm_backend_cuda.h"
#include "gm_error.h"
#include "gm_code_writer.h"
#include "gm_frontend.h"
#include "gm_transform_helper.h"
#include "gm_builtin.h"
#include "gm_argopts.h"

/*
void gm_cuda_gen::setTargetDir(const char* d) {
    assert(d != NULL);
    if (dname != NULL)
        delete[] dname;
    dname = new char[strlen(d) + 1];
    strcpy(dname, d);
}

void gm_cuda_gen::setFileName(const char* f) {
    assert(f != NULL);
    if (fname != NULL)
        delete[] fname;
    fname = new char[strlen(f) + 1];
    strcpy(fname, f);
}

bool gm_cuda_gen::do_local_optimize() {
    return 1;

}

bool gm_cuda_gen::do_local_optimize_lib() {
    return 1;
}

bool gm_cuda_gen::do_generate() {
    return 1;

}

void gm_cuda_gen::do_generate_begin() {

}

void gm_cuda_gen::do_generate_end() {

}
*/
/*void gm_cuda_gen::build_up_language_voca() {

}

void gm_cuda_gen::init_gen_steps() {

}

void gm_cuda_gen::open_output_files() {

}

void gm_cuda_gen::close_output_files(bool remove_files) {

}*/

void gm_cuda_gen::generate_lhs_id(ast_id* i) {

}

void gm_cuda_gen::generate_lhs_field(ast_field* i) {

}

void gm_cuda_gen::generate_rhs_id(ast_id* i) {

}

void gm_cuda_gen::generate_rhs_field(ast_field* i) {

}
/*
void gm_cuda_gen::generate_expr_foreign(ast_expr* e) {
*/
}

void gm_cuda_gen::generate_expr_builtin(ast_expr* i) {

}

void gm_cuda_gen::generate_expr_minmax(ast_expr* i) {

}

void gm_cuda_gen::generate_expr_abs(ast_expr* i) {

}

void gm_cuda_gen::generate_expr_nil(ast_expr* i) {

}

/*void gm_cuda_gen::generate_expr_type_conversion(ast_expr* e) {

}*/

/*const char* gm_cuda_gen::get_type_string(ast_typedecl* t) {

}*/

const char* gm_cuda_gen::get_type_string(int prim_t) {
    
    const char* p;
    return p;
}
/*
void gm_cuda_gen::generate_expr_list(std::list<ast_expr*>& L) {
}

void gm_cuda_gen::generate_expr(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_val(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_inf(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_uop(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_ter(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_bin(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_comp(ast_expr* e) {
}

bool gm_cuda_gen::check_need_para(int optype, int up_optype, bool is_right) {
    return 1;
}
*/

void gm_cuda_gen::generate_sent_nop(ast_nop* i) {

}

void gm_cuda_gen::generate_sent_reduce_assign(ast_assign* i) {

}

/*void gm_cuda_gen::generate_sent_defer_assign(ast_assign* i) {

}*/

void gm_cuda_gen::generate_sent_vardecl(ast_vardecl* i) {

}

void gm_cuda_gen::generate_sent_foreach(ast_foreach* i) {

}

void gm_cuda_gen::generate_sent_bfs(ast_bfs* i) {

}
/*
void gm_cuda_gen::generate_sent(ast_sent* i) {

}

void gm_cuda_gen::generate_sent_assign(ast_assign* i) {

}

void gm_cuda_gen::generate_sent_if(ast_if* i) {

}

void gm_cuda_gen::generate_sent_while(ast_while* i) {

}
*/
/*
void gm_cuda_gen::generate_sent_block(ast_sentblock* i) {

}

void gm_cuda_gen::generate_sent_block(ast_sentblock* i, bool need_br) {

}*/
/*
void gm_cuda_gen::generate_sent_return(ast_return* i) {

}

void gm_cuda_gen::generate_sent_call(ast_call* i) {

}

void gm_cuda_gen::generate_sent_foreign(ast_foreign* f) {

}*/
/*const char* gm_cuda_gen::get_function_name_map_reduce_assign(int reduceType) {
    char p[];
    return p;
}

void gm_cuda_gen::generate_sent_block_enter(ast_sentblock* b) {

}

void gm_cuda_gen::generate_sent_block_exit(ast_sentblock* b) {

}*/

/*void gm_cuda_gen::generate_idlist(ast_idlist* i) {

}

void gm_cuda_gen::generate_proc(ast_procdef* proc) {

}
void gm_cuda_gen::generate_mapaccess(ast_expr_mapaccess* e) {

} */

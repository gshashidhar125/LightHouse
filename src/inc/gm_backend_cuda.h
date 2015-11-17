#ifndef GM_BACKEND_CUDA
#define GM_BACKEND_CUDA

#include "gm_backend.h"
#include "gm_misc.h"
#include "gm_code_writer.h"
#include "gm_compile_step.h"
#include "gm_backend_cuda_opt_steps.h"
#include "gm_backend_cuda_gen_steps.h"

#include <list>

class scope;
class gm_cuda_gen: public gm_backend, public gm_code_generator {

public:
    gm_cuda_gen() :
        gm_code_generator(Body)/*, gm_code_generator(cudaBody)*/, fname(NULL), dname(NULL), f_header(NULL), f_body(NULL), f_cudaBody(NULL), insideCudaKernel(false), currentProc(NULL), currentScope(NULL), globalScope(NULL), GPUMemoryScope(NULL), printingMacro(false) {

        init();
    }

protected:
    void init() {
        //init_opt_steps();
        init_gen_steps();
        //build_up_language_voca();
        printf("Initialization");
    }

public:
    virtual ~gm_cuda_gen() {
        //close_output_files();
    }

    virtual void setTargetDir(const char* dname);
    virtual void setFileName(const char* fname);

    virtual bool do_local_optimize();
    virtual bool do_local_optimize_lib();
    virtual bool do_generate();

    virtual void do_generate_begin();
    virtual void do_generate_end();

    void printToFile(const char* s);
protected:
    std::list<gm_compile_step*> opt_steps;
    std::list<gm_compile_step*> gen_steps;

//    virtual void build_up_language_voca();
    virtual void init_opt_steps();
    virtual void init_gen_steps();

protected:
    char *fname;
    char *dname;

    gm_code_writer Header;
    gm_code_writer Body;
    gm_code_writer cudaBody;
    FILE *f_header;
    FILE *f_body;
    FILE *f_cudaBody;
    FILE *f_shell;

    ast_procdef* currentProc;
    bool insideCudaKernel;
    scope* currentScope;
    scope* globalScope;
    scope* GPUMemoryScope;
    bool printingMacro;

    bool open_output_files();
    void close_output_files(bool remove_files = false);

    virtual void add_include(const char* str1, gm_code_writer& Out, bool is_clib = true, const char* str2 = "");
    virtual void add_ifdef_protection(const char* str);
    //------------------------------------------------------------------------------
    // Generate Method from gm_code_generator
    //------------------------------------------------------------------------------
public:
    virtual void generate_rhs_id(ast_id* i);
    virtual void generate_rhs_field(ast_field* i);
    //virtual void generate_expr_foreign(ast_expr* e);
    virtual void generate_expr_builtin(ast_expr* e);
    virtual void generate_expr_minmax(ast_expr* e);
    virtual void generate_expr_abs(ast_expr* e);
    virtual void generate_expr_nil(ast_expr* e);
    //virtual void generate_expr_type_conversion(ast_expr* e);

    virtual const char* get_type_string(ast_typedecl* t);
    virtual const char* get_type_string(int prim_type);
    //virtual ast_typedecl* getNewTypeDecl(int typeId);
    virtual void generateMacroDefine(scope* s);
    void markGPUAndCPUGlobal();
    bool isOnGPUMemory(ast_id* i);

/*    virtual void generate_expr_list(std::list<ast_expr*>& L);
    virtual void generate_expr(ast_expr* e);
    virtual void generate_expr_val(ast_expr* e);
    virtual void generate_expr_inf(ast_expr* e);
    virtual void generate_expr_uop(ast_expr* e);
    virtual void generate_expr_ter(ast_expr* e);
    virtual void generate_expr_bin(ast_expr* e);
    virtual void generate_expr_comp(ast_expr* e);
    virtual bool check_need_para(int optype, int up_optype, bool is_right) {
        return gm_need_paranthesis(optype, up_optype, is_right);
    }*/

//    virtual void generate_mapaccess(ast_expr_mapaccess* e);


    virtual void generate_lhs_id(ast_id* i);
    virtual void generate_lhs_field(ast_field* i);
    virtual void generate_sent_nop(ast_nop* n);
    virtual void generate_sent_reduce_assign(ast_assign *a);
    virtual void generate_sent_defer_assign(ast_assign *a) {
        assert(false);
    } // should not be here
    virtual void generate_sent_vardecl(ast_vardecl *a);
    virtual void generate_sent_foreach(ast_foreach *a);
    virtual void generate_sent_bfs(ast_bfs* b);

/*    virtual void generate_sent(ast_sent* a);
    virtual void generate_sent_assign(ast_assign* a);
    virtual void generate_sent_if(ast_if* a);
    virtual void generate_sent_while(ast_while* a);
    //virtual void generate_sent_block(ast_sentblock *b);
    virtual void generate_sent_block(ast_sentblock* b, bool need_br);
    virtual void generate_sent_return(ast_return *r);*/
    virtual void generate_sent_call(ast_call* c);
/*    virtual void generate_sent_foreign(ast_foreign* f);
*/    //virtual const char* get_function_name_map_reduce_assign(int reduceType);

    //virtual void generate_sent_block_enter(ast_sentblock *b);
    //virtual void generate_sent_block_exit(ast_sentblock* b);

    virtual void generate_idlist(ast_idlist *i);
    virtual void generate_idlist_primitive(ast_idlist* idList);
    virtual void generate_lhs_default(int type);
    virtual void generate_proc(ast_procdef* proc);
    virtual void generate_kernel_function(ast_procdef* proc);
    virtual std::string generate_newKernelFunction(ast_foreach* f);
    virtual void generate_CudaAssignForIterator(ast_id* iter, bool isParallel);

    virtual std::string getNewTempVariable(ast_node* n);
    virtual void setGlobalScope(scope* s) {
        globalScope = s;
    }
    virtual scope* getGlobalScope() {   return globalScope; }
    
    virtual scope* getCurrentScope(){ return currentScope; }

    virtual void setCurrentScope(scope* s) {
        currentScope = s;
    }

    virtual scope* getGPUScope(){ return GPUMemoryScope; }

    virtual void setGPUScope(scope* s) {
        GPUMemoryScope = s;
    }
};

extern gm_cuda_gen CUDA_BE;
#endif

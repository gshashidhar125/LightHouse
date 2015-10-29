#include <stdio.h>
#include <string>
#include "gm_backend_cuda.h"
#include "gm_error.h"
#include "gm_code_writer.h"
#include "gm_frontend.h"
#include "gm_transform_helper.h"
#include "gm_builtin.h"
#include "gm_argopts.h"

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
/*
bool gm_cuda_gen::do_local_optimize() {
    return 1;

}

bool gm_cuda_gen::do_local_optimize_lib() {
    return 1;
}*/

bool gm_cuda_gen::do_generate() {

    if (!open_output_files()) return false;

    do_generate_begin();

    bool b = gm_apply_compiler_stage(this->gen_steps);
    if (b == false) {
        close_output_files(true);
        return false;
    }

    do_generate_end();
    close_output_files();
    return true;
}

void gm_cuda_gen::add_include(const char* string, gm_code_writer& Out, bool is_clib, const char* str2) {

    Out.push("#include ");
    if (is_clib)
        Out.push('<');
    else
        Out.push('"');
    Out.push(string);
    Out.push(str2);
    if (is_clib)
        Out.push('>');
    else
        Out.push('"');
    Out.NL();
}

void gm_cuda_gen::add_ifdef_protection(const char* s) {

    Header.push("#ifndef GM_GENERATED_CUDA_");
    Header.push_to_upper(s);
    Header.pushln("_H");
    Header.push("#define GM_GENERATED_CUDA_");
    Header.push_to_upper(s);
    Header.pushln("_H");
    Header.NL();
}

void gm_cuda_gen::do_generate_begin() {

    add_ifdef_protection(fname);
    add_include("stdio.h", Header);
    add_include("stdlib.h", Header);
    add_include("stdint.h", Header);
    add_include("float.h", Header);
    add_include("limits.h", Header);
    add_include("cmath", Header);
    add_include("algorithm", Header);

    add_include("cuda/cuda.h", Header);

    Header.NL();

    char temp[1024];
    sprintf(temp, "%s.h", fname);
    add_include(temp, Body, false);
    Body.NL();
}

void gm_cuda_gen::do_generate_end() {

    Header.NL();
    Header.pushln("#endif");
}

/*void gm_cuda_gen::build_up_language_voca() {

}*/

void gm_cuda_gen::init_gen_steps() {

    std::list<gm_compile_step*> &LIST = this->gen_steps;
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_gen_identify_par_regions));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_gen_proc));
}

enum memory {
    CPUMemory,
    GPUMemory,
};

class symbol {

public:
    symbol(ast_node* n, ast_typedecl* t) {
        node = n;
        type = t;
        parent = NULL;
        memLoc = CPUMemory;
        isIterator = false;
    }

    std::string name;
    ast_node* node;
    ast_typedecl* type;
    symbol* parent;
    std::list<symbol*> children;
    memory memLoc;
    bool isIterator;
    int iterType;

    symbol* getParent() {   return parent;  }
    void setParent(symbol* p) { parent = p; }

    memory getMemLoc() {    return memLoc;  }
    void setMemLoc(memory loc) {
        memLoc = loc;
    }

    int getType() { return type->get_typeid();  }
    ast_typedecl* getTypeDecl() { return type;  }
    char* getName() { 
        if (node->get_nodetype() == AST_ID) {
            return ((ast_id*)node)->get_orgname();   
        }
    }

    bool isSymbolIterator() {   return isIterator;  }
    void setSymbolIterator(bool s) {isIterator = s; }

    bool getSymbolIterType() {   return iterType;   }
    void setSymbolIteratorType(int s) {iterType = s; }

    symbol* checkAndAdd(ast_id* secondField) {

        for (std::list<symbol*>::iterator it = children.begin(); it !=  children.end(); it++) {
            if (strcmp((*it)->getName(), secondField->get_orgname()) == 0) {
                return *it;
            }
        }
        symbol* newSymbol = new symbol(secondField, secondField->getTypeInfo());
        children.push_back(newSymbol);
        return newSymbol;
    }
};

bool isSameFieldNames(ast_field* f1, ast_field* f2) {
    ast_id* id1 = f1->get_first();
    ast_id* id2 = f2->get_first();
    if (!(strcmp(id1->get_orgname(), id2->get_orgname()))) {
        id1 = f1->get_second();
        id2 = f2->get_second();
        if (!(strcmp(id1->get_orgname(), id2->get_orgname()))) {
            return true;
        }
    }
    return false;
}

bool sameVariableName(ast_node* n1, ast_node* n2) {

    if (n1->get_nodetype() != n2->get_nodetype())
        return false;
    if (n1->get_nodetype() == AST_ID) {
        ast_id* id1 = (ast_id*)n1;
        ast_id* id2 = (ast_id*)n2;
        return !(strcmp(id1->get_orgname(), id2->get_orgname()));
    }
    if (n1->get_nodetype() == AST_FIELD) {
        return isSameFieldNames((ast_field*)n1, (ast_field*)n2);
    }
    return false;
}

typedef std::list<ast_node*> listOfVariables;

bool findVariableInList(listOfVariables L, ast_node* n) {

    for (listOfVariables::iterator it = L.begin(); it != L.end(); it++) {

        ast_node* n2 = *it;
        if (sameVariableName(n, n2)) {
            return true;
        }
    }
    return false;
}

typedef std::map<ast_sent*, listOfVariables> mapForEachToVariables;
class gm_cuda_par_regions : public gm_apply {

public:
    bool insideForEachLoop;
    mapForEachToVariables forEachLoopVariables;
    listOfVariables currentList;
    std::list<symbol> symTable;

    gm_cuda_par_regions() {
        set_for_sent(true);
        set_separate_post_apply(true);
        insideForEachLoop = false;
        set_for_id(true);
    }

    bool apply(ast_procdef* proc) {
        /*std::list<ast_argdecl*>& inArgs = proc->get_in_args();
        std::list<ast_argdecl*>::iterator i;
        for (i = inArgs.begin(); i != inArgs.end(); i++) {
            ast_typedecl* type = (*i)->get_type();
            ast_idlist* idlist = (*i)->get_idlist();
            for (int ii = 0; ii < idlist->get_length(); ii++) {
                ast_id* id = idlist->get_item(ii);
                addToSymTable(id);
            }
        }
        std::list<ast_argdecl*>& outArgs = proc->get_out_args();
        for (i = outArgs.begin(); i != outArgs.end(); i++) {
            ast_typedecl* type = (*i)->get_type();
            ast_idlist* idlist = (*i)->get_idlist();
            for (int ii = 0; ii < idlist->get_length(); ii++) {
                ast_id* id = idlist->get_item(ii);
                addToSymTable(id);
            }
        }*/
        //ast_typedecl* returnType = proc->get_return_type();
        return true;
    }

    bool apply(ast_sent* s) {
        if (!insideForEachLoop) {
            if (s->get_nodetype() == AST_FOREACH) {
                ast_foreach* forEachStmt = (ast_foreach*)s;
                forEachStmt->set_sequential(false);
                traverse_up_invalidating(forEachStmt->get_parent());
            } // TODO: Can while loop be parallelized on GPU ?
            /*else if(s->get_nodetype() == AST_WHILE) {
            
            }*/
        }// else {
            /*switch(s->get_nodetype()) {

                case AST_VARDECL:   buildSymbolsDecl((ast_vardecl*)s);
                                    break;
                case AST_ASSIGN:    buildSymbolsAssign((ast_assign*)s);
                                    break;
                case AST_CALL:      buildSymbolsCall((ast_call*)s);
                                    break;
                case AST_FOREIGN:   buildSymbolsForeign((ast_foreign*)s);
                                    break;
                case AST_FOREACH:   buildSymbolsForEach((ast_foreach*)s);
                                    break;
                case AST_BFS:       buildSymbolsBFS((ast_bfs*)s);
                                    break;
            }*/
        //}
    }

    bool buildSymbolsDecl(ast_vardecl* s) {
        ast_idlist* idlist = s->get_idlist();
        ast_typedecl* type  = s->get_type();

       for (int i = 0; i < idlist->get_length(); i++) {
           ast_id* id = idlist->get_item(i);
           addToSymTable(id);
       }
    }

    symbol* findSymbol(const char* sym) {
        for (std::list<symbol>::iterator it = symTable.begin(); it != symTable.end(); it++) {
            if (strcmp((*it).getName(), sym) == 0)
                return &(*it);
        }
        return NULL;
    }

    void moveSymbolToGPU(ast_node* node) {

        if (node->get_nodetype() == AST_ID) {
            ast_id* id = (ast_id*)node;
            symbol* sym = findSymbol(id->get_orgname());
            sym->setMemLoc(GPUMemory);
        }else if (node->get_nodetype() == AST_FIELD) {
            ast_field* field = (ast_field*)node;
            symbol* firstField = findSymbol(field->get_first()->get_orgname());
            for (std::list<symbol*>::iterator it = firstField->children.begin(); it != firstField->children.end(); it++) {
                
                if (strcmp((*it)->getName(), field->get_second()->get_orgname()) == 0) {
                    (*it)->setMemLoc(GPUMemory);
                    break;
                }
            }
        }
    }

    void addToSymTable(ast_id* id) {

        symbol* sym = findSymbol(id->get_orgname());
        if (sym == NULL) {
            sym = new symbol(id, id->getTypeInfo());
            symTable.push_back(*sym);
        }
    }

    symbol* addToSymTableAndReturn(ast_id* id) {

        addToSymTable(id);
        return findSymbol(id->get_orgname());
    }

    void addFieldToSymTable(ast_field* f) {

        symbol* firstField = addToSymTableAndReturn(f->get_first());
        firstField->checkAndAdd(f->get_second());
    }

    void addExprToSymTable(ast_expr* e) {

        ast_expr* left = e->get_left_op();
        if (left == NULL)
            return;
        if (left->is_field()) {
            addFieldToSymTable(left->get_field());
        }else if (left->is_id()) {
            addToSymTable(left->get_id());
        }else if (left->is_biop() || left->is_comp() || left->is_terop()) {
            ast_expr* lLeft = left->get_left_op();
            ast_expr* rLeft = left->get_right_op();
            addExprToSymTable(lLeft);
            addExprToSymTable(rLeft);
        }else {
            if (left->get_left_op() != NULL) {
                ast_expr* lLeft = left->get_left_op();
                addExprToSymTable(lLeft);
            }
        }

        ast_expr* right = e->get_right_op();
        if (right == NULL)
            return;

        if (right->is_field()) {
            addFieldToSymTable(right->get_field());
        }else if (right->is_id()) {
            addToSymTable(right->get_id());
        }else if (right->is_biop() || right->is_comp() || right->is_terop()) {
            ast_expr* lRight = right->get_right_op();
            ast_expr* rRight = right->get_right_op();
            addExprToSymTable(lRight);
            addExprToSymTable(rRight);
        }else {
            if (right->get_left_op() != NULL) {
                ast_expr* lRight = right->get_left_op();
                addExprToSymTable(lRight);
            }
        }
    }

    bool buildSymbolsAssign(ast_assign* s) {
        if (s->get_assign_type() == GMASSIGN_NORMAL) {
            if (s->is_target_scalar()) {
                ast_id* id = s->get_lhs_scala();
                addToSymTable(id);
            }else if (s->is_target_field()) {
                ast_field* field = (ast_field *)s->get_lhs_field();
                addFieldToSymTable(field);
            }
        }else if (s->get_assign_type() == GMASSIGN_REDUCE) {

            ast_assign* reduceAssign = (ast_assign*)s;
            if (reduceAssign->is_target_scalar()) {
                ast_id* target = reduceAssign->get_lhs_scala();
                addToSymTable(target);
            } else if (reduceAssign->is_target_field()) {
                ast_field* targetField = reduceAssign->get_lhs_field();
                addFieldToSymTable(targetField);
            }
            ast_expr* rhs = reduceAssign->get_rhs();
            addExprToSymTable(rhs);
            ast_id* bound = reduceAssign->get_bound();
            addToSymTable(bound);
            if (reduceAssign->is_defer_assign()) {
                printf("Defer Assign\n");
            }
        }
    }

    bool buildSymbolsCall(ast_call* s) {

    }

    bool buildSymbolsForeign(ast_foreign* s) {

    }

    bool buildSymbolsForEach(ast_foreach* s) {

        ast_id* iterator = s->get_iterator();
        addToSymTable(iterator);

        if (s->is_source_field()) {
            ast_field* sourceField = s->get_source_field();
            addFieldToSymTable(sourceField);
        } else {
            ast_id* source = s->get_source();
            addToSymTable(source);
        }
        if (s->get_filter() != NULL) {
            addExprToSymTable(s->get_filter());
        }
    }

    bool buildSymbolsBFS(ast_bfs* s) {

    }

    void printSymbols() {

        printf("SymbolTable:\n");
        for (std::list<symbol>::iterator symIt = symTable.begin(); symIt != symTable.end(); symIt++) {
            printf("    %s, Type = %s   ", (*symIt).getName(), gm_get_type_string((*symIt).getType()));
            if ((*symIt).isSymbolIterator()) {
                symbol* parent = (*symIt).getParent();
                while (parent != NULL) {
                    
                    printf("-->%s ", parent->getName());
                    parent = parent->getParent();
                }
            }
            (*symIt).getTypeDecl()->dump_tree(0);
            /*if ((*symIt).getMemLoc() == GPUMemory)
                printf(", **GPU Memory**\n");
            else*/
                printf("\n");
            for (std::list<symbol*>::iterator childIt = (*symIt).children.begin(); childIt != (*symIt).children.end(); childIt++) {
                printf("        %s, Type = %s   ", (*childIt)->getName(), gm_get_type_string((*childIt)->getType()));
                (*childIt)->getTypeDecl()->dump_tree(0);
                /*if ((*childIt)->getMemLoc() == GPUMemory)
                    printf(", **GPU Memory**\n");
                else*/
                    printf("\n");
            }
        }
        printf("End Symbol Table\n");
    }

    void addIteratorForSymbol(ast_id* iterator, ast_id* source, int iterType) {
        symbol* iterSymbol = findSymbol(iterator->get_orgname());
        assert(iterSymbol != NULL);
        symbol* sourceSymbol = findSymbol(source->get_orgname());
        assert(sourceSymbol != NULL);

        iterSymbol->setParent(sourceSymbol);
        iterSymbol->setSymbolIteratorType(iterType);
        iterSymbol->setSymbolIterator(true);
    }

    bool apply(ast_id *s) {
        ast_node* parent = s->get_parent();
        if (insideForEachLoop == true) {
            if (parent->get_nodetype() == AST_FIELD) {
                if (((ast_field*)parent)->get_first() == s) {
                } else if (((ast_field*)parent)->get_second() == s) {
                    if (!findVariableInList(currentList, parent))
                        currentList.push_back(s->get_parent());
                    moveSymbolToGPU(parent);
                }
            } else if (parent->get_nodetype() == AST_MAPACCESS) {
                if (!findVariableInList(currentList, parent))
                    currentList.push_back(s->get_parent());
            } else {
                if (!findVariableInList(currentList, s))
                    currentList.push_back(s);
                moveSymbolToGPU(s);
            }
        }
        if (parent == NULL)
            return true;
        if (parent->get_nodetype() == AST_FIELD) {
            if (((ast_field*)parent)->get_first() == s) {
            } else if (((ast_field*)parent)->get_second() == s) {
                addFieldToSymTable((ast_field*)parent);
            }
        } else if (parent->get_nodetype() == AST_MAPACCESS) {
        } else {
            addToSymTable(s);
        }
    }

    bool apply2(ast_sent* s) {
        if (!insideForEachLoop && s->get_nodetype() == AST_FOREACH) {
            ast_foreach* forEachStmt = (ast_foreach*)s;
            if (forEachStmt->is_parallel()) {
                insideForEachLoop = true;
                set_for_sent(false);
                set_separate_post_apply(false);
                
                listOfVariables *variablesList = new listOfVariables();
                
                if (forEachStmt->get_iterator() != NULL) {
                    moveSymbolToGPU(forEachStmt->get_iterator());
                    if (!findVariableInList(currentList, forEachStmt->get_iterator()))
                        currentList.push_back(forEachStmt->get_iterator());
                }
                if (forEachStmt->get_source() != NULL) {
                    moveSymbolToGPU(forEachStmt->get_source());
                    if (!findVariableInList(currentList, forEachStmt->get_source()))
                        currentList.push_back(forEachStmt->get_source());
                }
                if (forEachStmt->get_source2() != NULL) {
                    moveSymbolToGPU(forEachStmt->get_source2());
                    if (!findVariableInList(currentList, forEachStmt->get_source2()))
                        currentList.push_back(forEachStmt->get_source2());
                }
                if (forEachStmt->is_source_field()) {
                    moveSymbolToGPU(forEachStmt->get_source_field());
                }
                if (forEachStmt->get_filter() != NULL) {
                    forEachStmt->get_filter()->traverse_pre(this);
                }

                forEachStmt->get_body()->traverse_pre(this);

                while (!currentList.empty()) {
                    variablesList->push_back(currentList.front());
                    currentList.pop_front();
                }

                std::pair<ast_sent*, listOfVariables> t(s, *variablesList);
                forEachLoopVariables.insert(t);
                
                set_for_sent(true);
                set_separate_post_apply(true);
                insideForEachLoop = false;
            }
        }
        if (s->get_nodetype() == AST_FOREACH) {
            ast_foreach* forEachStmt = (ast_foreach*)s;
            ast_id* iterator = forEachStmt->get_iterator();
            ast_id* source = forEachStmt->get_source();
            addIteratorForSymbol(iterator, source, forEachStmt->get_iter_type());
        }
    }

    void traverse_up_invalidating(ast_node* s) {
        if (s == NULL)
            return;
        if (s->get_nodetype() == AST_FOREACH) {
            ast_foreach* forEachStmt = (ast_foreach*)s;
            forEachStmt->set_sequential(true);
            traverse_up_invalidating(forEachStmt->get_parent());
        }
        traverse_up_invalidating(s->get_parent());
    }

    void printVariablesInsideForEachLoop() {

        mapForEachToVariables::iterator it;
        for (it = forEachLoopVariables.begin(); it != forEachLoopVariables.end();
                it++) {

            listOfVariables L = (*it).second;
            /*for (listOfVariables::iterator iit = L.begin(); iit != L.end(); iit++) {
                ast_node* n1 = *iit;
                listOfVariables::iterator iit2 = iit;
                iit2++;
                for (; iit2 != L.end(); iit2++) {
                    ast_node* n2 = *iit2;
                    if (n1->get_nodetype() == AST_ID &&
                        n2->get_nodetype() == AST_ID) {
                        ast_id* id1 = (ast_id*)n1;
                        ast_id* id2 = (ast_id*)n2;
                        if (!(strcmp(id1->get_orgname(), id2->get_orgname()))) {
                            iit2 = L.erase(iit2);
                        }
                    } else if (n1->get_nodetype() == AST_FIELD &&
                               n2->get_nodetype() == AST_FIELD) {
                    
                        ast_field* f1 = (ast_field*)n1;
                        ast_field* f2 = (ast_field*)n2;
                        if (isSameFieldNames(f1, f2)) {
                            iit2 = L.erase(iit2);
                        }
                    }
                }
            }*/
            printf("Next ForEachLoop:\n");
            for (listOfVariables::iterator iit = L.begin(); iit != L.end(); 
                    iit++) {
                ast_node *node = *iit;
                if (node->get_nodetype() == AST_FIELD) {
                    ast_field* fieldNode = (ast_field*)node;
                    printf("    AST_FIELD:: %s.%s\n", fieldNode->get_first()->get_orgname(), fieldNode->get_second()->get_orgname());
                } else if (node->get_nodetype() == AST_MAPACCESS) {
                    ast_mapaccess* mapNode = (ast_mapaccess*)node;
                    printf("    AST_MAPACCESS:: %s[]\n", mapNode->get_map_id()->get_orgname());
                } else if (node->get_nodetype() == AST_ID){
                    ast_id* idNode = (ast_id*) node;
                    printf("    AST_ID:: %s\n", idNode->get_orgname());
                }
            }
        }
    }

private:
    
};

/* Delete This
typedef std::pair<ast_node*, ast_node*> depEdge;
class gm_cuda_dependency_graph : public gm_apply {

public:
    std::map<std::string, ast_node*> symTable;
    gm_cuda_dependency_graph() {
        set_for_sent(true);
        set_for_id(true);
    }

    bool apply(ast_sent* s) {

    }

    bool apply(ast_id* s) {
        ast_node* parent = s->get_parent();
        if (parent == NULL) {
        
            if (symTable.find(s->get_orgname()) == symTable.end()) {
                symTable.insert(std::pair<std::string, ast_node*>(s->get_orgname(), s));
            }
            return true;
        }
        if (parent->get_nodetype() == AST_FIELD) {
            ast_field* fieldNode = (ast_field*) parent;
            std::string varName(fieldNode->get_first()->get_orgname());
            varName += s->get_orgname();
            if (symTable.find(varName) == symTable.end()) {
                symTable.insert(std::pair<std::string, ast_node*>(varName, parent));
            } else if (parent->get_nodetype() == AST_MAPACCESS) {
                std::string varName2(fieldNode->get_first()->get_orgname());
                varName2 += s->get_orgname();
                if (symTable.find(varName2) == symTable.end()) {
                    symTable.insert(std::pair<std::string, ast_node*>(varName2, parent));
                }
            }
        }
    }
};*/

    gm_cuda_par_regions transfer;

void gm_cuda_gen_identify_par_regions::process(ast_procdef* proc) {
    
    //gm_cuda_dependency_graph dep_graph;
    
    transfer.set_for_proc(true);
    proc->traverse_pre(&transfer);
    transfer.set_for_proc(false);
    printf("\n\n----------Ended Pre Traversal-----------\n\n");
    proc->traverse_post(&transfer);
    proc->dump_tree(0);
    transfer.printSymbols();
    printf("Variables list\n---------------\n");
    transfer.printVariablesInsideForEachLoop(); 

    //proc->traverse_pre(&dep_graph);
}

void gm_cuda_gen_proc::process(ast_procdef* proc) {
    CUDA_BE.generate_proc(proc);
}

bool gm_cuda_gen::open_output_files() {
    char temp[1024];
    assert(dname != NULL);
    assert(fname != NULL);

    sprintf(temp, "%s/%s.h", dname, fname);
    f_header = fopen(temp, "w");
    if (f_header == NULL) {
        gm_backend_error(GM_ERROR_FILEWRITE_ERROR, temp);
        return false;
    }
    Header.set_output_file(f_header);

    sprintf(temp, "%s/%s.cpp", dname, fname);
    f_body = fopen(temp, "w");
    if (f_body == NULL) {
        gm_backend_error(GM_ERROR_FILEWRITE_ERROR, temp);
        return false;
    }
    Body.set_output_file(f_body);
    //get_lib()->set_code_writer(&Body);
    return true;
}

void gm_cuda_gen::close_output_files(bool remove_files) {

    char temp[1024];
    if (f_header != NULL) {
        Header.flush();
        fclose(f_header);
        if (remove_files) {
            sprintf(temp, "rm %s/%s.h", dname, fname);
            system(temp);
        }
        f_header = NULL;
    }
    if (f_body != NULL) {
        Body.flush();
        fclose(f_body);
        if(remove_files) {
            sprintf(temp, "rm %s/%s.cpp", dname, fname);
            system(temp);
        }
        f_body = NULL;
    }
}

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

}*/

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

}*/

void gm_cuda_gen::generate_proc(ast_procdef* proc) {

    printf("Inside Code Generation of CUDA\n");

    // Parallel Regions Analysis
    gm_cuda_par_regions PRA = transfer;
    generate_kernel_function(proc);
}

// Generate the Kernel call definition
void gm_cuda_gen::generate_kernel_function(ast_procdef* proc) {

    gm_code_writer& Out = Body;
    std::string str("__global__ void ");

    str = str + proc->get_procname()->get_genname();
    Out.push(str.c_str());

    Out.push("(");
    
    Out.NL();
}

/*
void gm_cuda_gen::generate_mapaccess(ast_expr_mapaccess* e) {

} */

#include <stdio.h>
#include <string>
#include "gm_backend_cuda.h"
#include "gm_error.h"
#include "gm_code_writer.h"
#include "gm_frontend.h"
#include "gm_transform_helper.h"
#include "gm_builtin.h"
#include "gm_argopts.h"

bool doPrint = true;

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

void gm_cuda_gen::printToFile(const char* s) {
    if (insideCudaKernel)
        cudaBody.push(s);
    else
        Body.push(s);
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

    do_generate_user_main();
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
    //add_include("limits.h", Header);
    //add_include("cmath", Header);
    //add_include("algorithm", Header);

    //add_include("cuda/cuda.h", Header);

    Header.NL();

    char temp[1024];
    sprintf(temp, "%s.h", fname);
    add_include(temp, Body, false);
    sprintf(temp, "%s.cu", fname);
    add_include(temp, Body, false);
    Body.pushln("#include <fstream>");
    
    /*sprintf(temp, "GlobalBarrier.cuh");
    add_include(temp, cudaBody, false);*/
    
    Body.NL();
    cudaBody.NL();
    cudaBody.flush();
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
        isParallelIterator = false;
    }

    std::string codeGenName;
    std::string startIndexStr, endIndexStr;
    std::string numOfThreads;
    ast_node* node;
    ast_typedecl* type;
    symbol* parent;
    std::list<symbol*> children;
    memory memLoc;
    bool isIterator, isParallelIterator;
    int iterType;

    std::string getCodeGenName() {
        return codeGenName;
    }
    void setCodeGenName(std::string n) {
        codeGenName = n;
    }

    std::string getStartIndexStr() {
        assert(isIterator && !isParallelIterator);
        return startIndexStr;
    }

    std::string getEndIndexStr() {
        assert(isIterator && !isParallelIterator);
        return endIndexStr;
    }

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

    bool isSymbolParallelIterator() {   return isParallelIterator;  }
    void setSymbolParallelIterator(bool s) {isParallelIterator = s; }

    int getSymbolIterType() {   return iterType;   }
    void setSymbolIteratorType(int s) {iterType = s; }

    std::string getIteratorTypeString() {
        assert(isIterator);
        std::string typeString;
        if (type->is_node_iterator() || type->is_edge_iterator()) {
            typeString = "int";
        } else if (gm_is_collection_iterator_type(iterType)) {
            typeString = "collection";
        }
        return typeString;
    }

    symbol* checkAndAdd(ast_id* secondField) {

        for (std::list<symbol*>::iterator it = children.begin(); it !=  children.end(); it++) {
            if (strcmp((*it)->getName(), secondField->get_orgname()) == 0) {
                return *it;
            }
        }
        symbol* newSymbol = new symbol(secondField, secondField->getTypeInfo());
        children.push_back(newSymbol);
        //newSymbol->setParent(this);
        return newSymbol;
    }

    std::string getNumOfThreads() {
        getIndexParallel();
        return numOfThreads;
    }

    std::string getIndexParallel() {
        std::string indexStr;
        symbol* parentSym = getParent();

        if (parentSym->type->is_graph()) {
            if (gm_is_all_graph_node_iteration(iterType)) {
                indexStr = "tId";
                numOfThreads = "NumNodes + 1";
            } else if (gm_is_all_graph_edge_iteration(iterType)) {
                indexStr = "tId";
                numOfThreads = "NumEdges";
            }
        } else if (parentSym->type->is_node()) {
            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_NODE_NBRS:
                        indexStr = "G1[G0[" + temp + "] + tId]";
                        numOfThreads = "G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_IN_NBRS:
                        indexStr = "IN-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "IN-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_UP_NBRS:
                        indexStr = "UP-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "Up-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_DOWN_NBRS:
                        indexStr = "DOWN-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "DOWN-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
            }
        } else if (parentSym->type->is_edge()) {
            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_EDGE_NBRS:
                        indexStr = "Edge NBRS";
                        numOfThreads = "Edge NBRS";
                        break;
                case GMITER_EDGE_IN_NBRS:
                        indexStr = "Edge IN-Nbrs";
                        numOfThreads = "Edge IN-Nbrs";
                        break;
                case GMITER_EDGE_UP_NBRS:
                        indexStr = "Edge UP-Nbrs";
                        numOfThreads = "Edge UP-Nbrs";
                        break;
                case GMITER_EDGE_DOWN_NBRS:
                        indexStr = "Edge DOWN-Nbrs";
                        numOfThreads = "Edge DOWN-Nbrs";
                        break;
            }
        } else if (parentSym->type->is_node_iterator()) {

            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_NODE_NBRS:
                        indexStr = "G1[G0[" + temp + "] + tId]";
                        numOfThreads = "G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_IN_NBRS:
                        indexStr = "IN-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "IN-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_UP_NBRS:
                        indexStr = "UP-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "Up-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
                case GMITER_NODE_DOWN_NBRS:
                        indexStr = "DOWN-Nbrs G1[G0[" + temp + "] + tId]";
                        numOfThreads = "DOWN-Nbrs G0[" + temp + " + 1] - G0[" + temp + "]";
                        break;
            }
        } else if (parentSym->type->is_edge_iterator()) {

            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_EDGE_NBRS:
                        indexStr = "Edge NBRS";
                        numOfThreads = "Edge NBRS";
                        break;
                case GMITER_EDGE_IN_NBRS:
                        indexStr = "Edge IN-Nbrs";
                        numOfThreads = "Edge IN-Nbrs";
                        break;
                case GMITER_EDGE_UP_NBRS:
                        indexStr = "Edge UP-Nbrs";
                        numOfThreads = "Edge UP-Nbrs";
                        break;
                case GMITER_EDGE_DOWN_NBRS:
                        indexStr = "Edge DOWN-Nbrs";
                        numOfThreads = "Edge DOWN-Nbrs";
                        break;
            }
        }
        setCodeGenName(indexStr);
        return indexStr;
    }

    std::string getIndexSequential() {
        //std::string startIndexStr, endIndexStr;
        symbol* parentSym = getParent();

        if (parentSym->type->is_graph()) {
            if (gm_is_all_graph_node_iteration(iterType)) {
                startIndexStr = "0";
                endIndexStr = "NumNodes";
            } else if (gm_is_all_graph_edge_iteration(iterType)) {
                startIndexStr = "0";
                endIndexStr = "NumEdges";
            }
        } else if (parentSym->type->is_node()) {
            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_NODE_NBRS:
                        startIndexStr = "G0[" + temp + "]";
                        endIndexStr = "G0[" + temp + " + 1]";
                        break;
                case GMITER_NODE_IN_NBRS:
                        startIndexStr = "IN-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "IN-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
                case GMITER_NODE_UP_NBRS:
                        startIndexStr = "UP-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "UP-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
                case GMITER_NODE_DOWN_NBRS:
                        startIndexStr = "DOWN-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "DOWN-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
            }
        } else if (parentSym->type->is_edge()) {
            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_EDGE_NBRS:
                        startIndexStr = "Edge NBRS";
                        endIndexStr = "Edge NBRS";
                        break;
                case GMITER_EDGE_IN_NBRS:
                        startIndexStr = "Edge IN-Nbrs";
                        endIndexStr = "Edge IN-Nbrs";
                        break;
                case GMITER_EDGE_UP_NBRS:
                        startIndexStr = "Edge UP-Nbrs";
                        endIndexStr = "Edge UP-Nbrs";
                        break;
                case GMITER_EDGE_DOWN_NBRS:
                        startIndexStr = "Edge DOWN-Nbrs";
                        endIndexStr = "Edge DOWN-Nbrs";
                        break;
            }
        } else if (parentSym->type->is_node_iterator()) {

            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_NODE_NBRS:
                        startIndexStr = "G0[" + temp + "]";
                        endIndexStr = "G0[" + temp + " + 1]";
                        break;
                case GMITER_NODE_IN_NBRS:
                        startIndexStr = "IN-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "IN-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
                case GMITER_NODE_UP_NBRS:
                        startIndexStr = "UP-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "UP-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
                case GMITER_NODE_DOWN_NBRS:
                        startIndexStr = "DOWN-Nbrs G1[G0[" + temp + "]]";
                        endIndexStr = "DOWN-Nbrs G1[G0[" + temp + " + 1]]";
                        break;
            }
        } else if (parentSym->type->is_edge_iterator()) {

            std::string temp = parentSym->getName();
            switch(iterType) {
                case GMITER_EDGE_NBRS:
                        startIndexStr = "Edge NBRS";
                        break;
                case GMITER_EDGE_IN_NBRS:
                        startIndexStr = "Edge IN-Nbrs";
                        break;
                case GMITER_EDGE_UP_NBRS:
                        startIndexStr = "Edge UP-Nbrs";
                        break;
                case GMITER_EDGE_DOWN_NBRS:
                        startIndexStr = "Edge DOWN-Nbrs";
                        break;
            }
        }
        return startIndexStr;
    }
};

ast_typedecl* getNewTypeDecl(int typeId) {

    if (gm_is_prim_type(typeId)) {
        return ast_typedecl::new_primtype(typeId);
    } else if (gm_is_nodeedge_type(typeId)) {
        return ast_typedecl::new_primtype(GMTYPE_INT);
    } else if (gm_is_node_iterator_type(typeId)) {
        return ast_typedecl::new_primtype(GMTYPE_INT);
    } else if (gm_is_edge_iterator_type(typeId)){
        return ast_typedecl::new_primtype(GMTYPE_INT);
    } else if (gm_is_graph_type(typeId)){
        return ast_typedecl::new_graphtype(typeId);
    } else  if (gm_is_property_type(typeId)){
        return ast_typedecl::new_primtype(GMTYPE_INT);
    } else {
        // Not expected any other type.
        assert(false);
    }
    return NULL;
}

enum SCOPE_T {
    THREAD_LOCAL,
    //SHARED_MEMORY,
    GLOBAL_GPU_MEMORY,
    CPU_MEMORY,
    NO_SCOPE,
};

//typedef std::pair<ast_vardecl*, SCOPE_T>scope_entry;
class scope_entry {

public:
    scope_entry(ast_id* i, SCOPE_T s, ast_typedecl* t) : 
        id(i), memLoc(s), type(t) {}

    ast_id* getId() { return id; }
    const char* getName()   { return id->get_orgname(); }
    SCOPE_T getMemLoc() { return memLoc;    }
    void setMemLoc(SCOPE_T m) { 
        if (m == THREAD_LOCAL) {
            if (memLoc == GLOBAL_GPU_MEMORY)
                return;
            else if (memLoc == CPU_MEMORY) {
                memLoc = GLOBAL_GPU_MEMORY;
                return;
            }
        } else if (m == CPU_MEMORY) {
            if (memLoc == THREAD_LOCAL || memLoc == GLOBAL_GPU_MEMORY) {
                memLoc = GLOBAL_GPU_MEMORY;
                return;
            }
        }
        memLoc = m; 
    }
    int getType()   {   return type->get_typeid();  }
    ast_typedecl* getTypeDecl() { return type;  }
protected:
    ast_id* id;
    SCOPE_T memLoc;
    ast_typedecl* type;
};

class scope {

public:
    scope(ast_sent* stmt, ast_procdef* proc) : enclosingStmt(stmt), enclosingProc(proc), parent(NULL), forEachScope(true), localExprValue(0) { }
    scope(ast_procdef* proc) : enclosingProc(proc), parent(NULL), forEachScope(false), localExprValue(0) {}

    bool isForEachScope() {
        return forEachScope;
    }
    void setParentScope(scope* p) {
        parent = p;
    }

    std::list<scope_entry*> getScopeEntries() {
        return variables;
    }

    std::list<ast_vardecl*> getVarDecls() {
        return varDecls;
    }

    scope* getParentScope() {  return parent;  }
    void setGPUScope(scope* g) {
        gpuScope = g;
    }

    void setMacroName(std::string n) { name = n;  }
    std::string getMacroName()  { return name;  }

    int getLocalExprValue() {   return localExprValue;  }
    void setLocalExprValue(int l) {
        localExprValue = l;
    }

    void addVariableToScope(ast_id* varId) {
        if (varId->getSymInfo() == NULL)
            return;
        int typeId = getNewTypeDecl(varId->getTypeSummary())->get_typeid();
        addVariableToScope(varId, typeId);
    }

    void addVariableToScope(ast_id* varId, int typeId) {
        std::list<scope_entry*>::iterator varDeclIt = variables.begin();
        bool insideCudaKernel = false;
        ast_typedecl* varType = getNewTypeDecl(typeId);
        if (parent == NULL)
            insideCudaKernel = false;
        else
            insideCudaKernel = true;

        for (; varDeclIt != variables.end(); varDeclIt++) {
            scope_entry* var = *varDeclIt;
            if (!strcmp(varId->get_orgname(), var->getName())) {
                if (var->getType() != typeId) {
                    // Type mismatch
                    assert(false);
                }
                if (insideCudaKernel) {
                    var->setMemLoc(THREAD_LOCAL);
                    /*if (var->getMemLoc() == GLOBAL_GPU_MEMORY) {
                        gpuScope->variables.push_back(var);
                        //variables.erase(varDeclIt);
                    }*/
                } else {
                    var->setMemLoc(CPU_MEMORY);
                    /*if (var->getMemLoc() == GLOBAL_GPU_MEMORY) {
                        gpuScope->variables.push_back(var);
                        //variables.erase(varDeclIt);
                    }*/
                }
                return;
            }
        }

        if (parent != NULL) {
            varDeclIt = parent->variables.begin();
            for (; varDeclIt != parent->variables.end(); varDeclIt++) {
                scope_entry* var = *varDeclIt;
                if (!strcmp(varId->get_orgname(), var->getName())) {
                    if (var->getType() != typeId) {
                        // Type mismatch
                        assert(false);
                    }
                    var->setMemLoc(GLOBAL_GPU_MEMORY);
                    return;
                }
            }
        }
        scope_entry* newVar;
        if (insideCudaKernel)
            newVar = new scope_entry(varId, THREAD_LOCAL, varType);
        else
            newVar = new scope_entry(varId, CPU_MEMORY, varType);
        variables.push_back(newVar);

        /*ast_idlist* correctListToAdd = NULL;
        for (; varDeclIt != variables.end(); varDeclIt++) {
            ast_vardecl* vars = (*varDeclIt).first;
            ast_idlist* varList = vars->get_idlist();
            if (varList->contains(varId->get_orgname()) == true) {
                if (vars->get_type()->get_typeid() == typeId) {
                    // Already the variable is inserted to the scope
                    //assert(correctListToAdd == NULL);
                    return;
                } else {
                    // The same variable is declared as different type
                    assert(false);
                }
            } else if (vars->get_type()->get_typeid() == typeId) { //&&
                //       (*varDeclIt).second == currentScope
                //assert(correctListToAdd == NULL);
                correctListToAdd = varList;
            }
        }

        // Search in the Parent Scope
        if (parent != NULL) {
            varDeclIt = parent->variables.begin();
            for (; varDeclIt != parent->variables.end(); varDeclIt++) {
                ast_vardecl* vars = (*varDeclIt).first;
                ast_idlist* varList = vars->get_idlist();
                if (varList->contains(varId->get_orgname()) == true) {
                    if (vars->get_type()->get_typeid() == typeId) {
                        // Already the variable is inserted in the
                        // Global scope
                        //assert(correctListToAdd == NULL);
                        return;
                    } else {
                        // The same variable is declared as different type
                        assert(false);
                    }
                }// else if (vars->get_type()->get_typeid() == typeId) {
                 //   assert(correctListToAdd == NULL);
                 //   correctListToAdd = varList;
                //}
            }
        }
        if (correctListToAdd == NULL) {
            ast_typedecl* newType = getNewTypeDecl(typeId);
            ast_vardecl* newVarDecl = ast_vardecl::new_vardecl(newType, varId);

            scope_entry newScopeEntry(newVarDecl, currentScope);
            variables.push_back(newScopeEntry);
            return;
        }
            
        correctListToAdd->add_id(varId);*/
    }

    void addVariableDeclaration(ast_id* varId, ast_typedecl* varType) {
        
        ast_idlist* correctListToAdd = NULL;
        std::list<ast_vardecl*>::iterator varDeclIt = varDecls.begin();
        for (; varDeclIt != varDecls.end(); varDeclIt++) {
            ast_idlist* varList = (*varDeclIt)->get_idlist();
            if (varList->contains(varId->get_orgname()) == true) {
                if ((*varDeclIt)->get_type()->get_typeid() == varType->get_typeid()) {
                    // Already the variable is inserted in the
                    // declarations
                    return;
                } else {
                    ast_typedecl* convertedType = getNewTypeDecl((*varDeclIt)->get_type()->get_typeid());
                    // Check if the type has been converted to primtype for CUDA.
                    if (convertedType->get_typeid() != varType->get_typeid()) {
                        // The same variable is declared as different type
                        assert(false);
                    }
                    return;
                }
            } else if ((*varDeclIt)->get_type()->get_typeid() == varType->get_typeid()) { //&&
                //       (*varDeclIt).second == currentScope
                //assert(correctListToAdd == NULL);
                if (varType->is_property()) {
                    if ((*varDeclIt)->get_type()->is_property() &&
                        (varType->getTargetTypeSummary() == (*varDeclIt)->get_type()->getTargetTypeSummary())) {

                        correctListToAdd = varList;
                    }
                } else {
                    correctListToAdd = varList;
                }
            }
        }
        

        if (correctListToAdd == NULL) {
            ast_vardecl* newVarDecl = ast_vardecl::new_vardecl(varType, varId);

            varDecls.push_back(newVarDecl);
            return;
        }
            
            printf("New Variable Added to Scope = %s\n", varId->get_orgname());
            varType->dump_tree(2);
            printf("Ended Scope add\n");
        correctListToAdd->add_id(varId);
    }

    void printScopeVariables() {

        std::list<scope_entry*>::iterator varDeclIt = variables.begin();
        for (; varDeclIt != variables.end(); varDeclIt++) {
            ast_typedecl* varType = (*varDeclIt)->getTypeDecl();
            printf("Type = ");
            varType->dump_tree(2);
            printf("\nVariables = %s, scope = %d\n", (*varDeclIt)->getName(), (*varDeclIt)->getMemLoc());
        }
    }

protected:
    ast_sent* enclosingStmt;
    ast_procdef* enclosingProc;
    scope* parent;
    scope* gpuScope;
    std::string name;
    std::list<scope_entry*> variables;
    std::list<ast_vardecl*> varDecls;
    bool forEachScope;
    int localExprValue;
};

bool gm_cuda_gen::isOnGPUMemory(ast_id* i) {
    
    std::list<scope_entry*> entries = currentScope->getScopeEntries();
    std::list<scope_entry*>::iterator varDeclIt = entries.begin();
    for (; varDeclIt != entries.end(); varDeclIt++) {
        scope_entry* var = *varDeclIt;
        if (!strcmp(i->get_orgname(), var->getName()))
            if (var->getMemLoc() == GLOBAL_GPU_MEMORY)
                return true;
    }
    entries = globalScope->getScopeEntries();
    varDeclIt = entries.begin();
    for (; varDeclIt != entries.end(); varDeclIt++) {
        scope_entry* var = *varDeclIt;
        if (!strcmp(i->get_orgname(), var->getName()))
            if (var->getMemLoc() == GLOBAL_GPU_MEMORY)
                return true;
    }
    entries = GPUMemoryScope->getScopeEntries();
    varDeclIt = entries.begin();
    for (; varDeclIt != entries.end(); varDeclIt++) {
        scope_entry* var = *varDeclIt;
        if (!strcmp(i->get_orgname(), var->getName()))
            if (var->getMemLoc() == GLOBAL_GPU_MEMORY)
                return true;
    }
    return false;
}

void gm_cuda_gen::markGPUAndCPUGlobal() {

    // Add variables to global scope declaration
    std::list<scope_entry*> varList = GPUMemoryScope->getScopeEntries();
    std::list<scope_entry*>::iterator varListIt = varList.begin();
    for (; varListIt != varList.end(); varListIt++) {
        ast_id* varId = (*varListIt)->getId();
        ast_typedecl* varType = (*varListIt)->getTypeDecl();
        GPUMemoryScope->addVariableDeclaration(varId, varType);
    }
    varList = globalScope->getScopeEntries();
    varListIt = varList.begin();
    for (; varListIt != varList.end(); varListIt++) {
        ast_id* varId = (*varListIt)->getId();
        ast_typedecl* varType = (*varListIt)->getTypeDecl();
        if ((*varListIt)->getMemLoc() == GLOBAL_GPU_MEMORY) {
            GPUMemoryScope->addVariableDeclaration(varId, varType);
            ast_id* newVarId = varId->copy(false);
            std::string str = std::string("h_") + varId->get_orgname();
            newVarId->set_orgname(str.c_str());
            globalScope->addVariableDeclaration(newVarId, varType);
        }
        else
            globalScope->addVariableDeclaration(varId, varType);
    }
    if (currentScope == globalScope)
        return;

    varList = currentScope->getScopeEntries();
    varListIt = varList.begin();
    for (; varListIt != varList.end(); varListIt++) {
        ast_id* varId = (*varListIt)->getId();
        ast_typedecl* varType = (*varListIt)->getTypeDecl();
        if ((*varListIt)->getMemLoc() == GLOBAL_GPU_MEMORY)
            GPUMemoryScope->addVariableDeclaration(varId, varType);
        else
            globalScope->addVariableDeclaration(varId, varType);
    }
}

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
    bool postTraversal, insideForEachLoop;
    mapForEachToVariables forEachLoopVariables;
    listOfVariables currentList;
    std::list<symbol> symTable;

    gm_cuda_par_regions() {
        set_for_sent(true);
        set_separate_post_apply(true);
        postTraversal = false;
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
        if (!postTraversal) {
            if (s->get_nodetype() == AST_FOREACH) {
                ast_foreach* forEachStmt = (ast_foreach*)s;
                if (insideForEachLoop == false) {
                    insideForEachLoop = true;
                    forEachStmt->set_sequential(false);
                    //traverse_up_invalidating(forEachStmt->get_parent());
                } else {
                    forEachStmt->set_sequential(true);
                }
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
            if(doPrint)
            (*symIt).getTypeDecl()->dump_tree(0);
            if ((*symIt).getMemLoc() == GPUMemory)
                printf(", **GPU Memory**\n");
            else
                printf("\n");
            for (std::list<symbol*>::iterator childIt = (*symIt).children.begin(); childIt != (*symIt).children.end(); childIt++) {
                printf("        %s, Type = %s   ", (*childIt)->getName(), gm_get_type_string((*childIt)->getType()));
                if(doPrint)
                (*childIt)->getTypeDecl()->dump_tree(0);
                if ((*childIt)->getMemLoc() == GPUMemory)
                    printf(", **GPU Memory**\n");
                else
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
        if (postTraversal == true) {
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
        if (!postTraversal && s->get_nodetype() == AST_FOREACH) {
            ast_foreach* forEachStmt = (ast_foreach*)s;
            if (forEachStmt->is_parallel()) {
                postTraversal = true;
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
                postTraversal = false;
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
    
    if (!OPTIONS.get_arg_bool(GMARGFLAG_DUMP_TREE)) {
        doPrint = false;
    }
    printf("Do Print = %d\n", doPrint);
    /*transfer.set_for_proc(true);
    proc->traverse_pre(&transfer);
    transfer.set_for_proc(false);
    printf("\n\n----------Ended Pre Traversal-----------\n\n");
    proc->traverse_post(&transfer);*/
    proc->traverse_both(&transfer);

    if(doPrint)
        proc->dump_tree(0);

    transfer.printSymbols();
    printf("Variables list\n---------------\n");
    transfer.printVariablesInsideForEachLoop(); 

    //proc->traverse_pre(&dep_graph);
}

class findDeviceVariablesInExpr : public gm_apply {

public:
    std::list<ast_id*> deviceVariables;
    std::list<ast_field*> deviceFields;
    bool insideCudaKernel, printingMacro;

    findDeviceVariablesInExpr(bool i, bool j) {
        set_for_sent(false);
        set_for_id(true);
        set_for_expr(true);
        insideCudaKernel = i;
        printingMacro = j;
    }
    std::list<ast_id*> getVars() {  return deviceVariables;}
    std::list<ast_field*> getFields() {  return deviceFields;}
    bool apply(ast_sent* stmt) {
        if (stmt->get_nodetype() == AST_ASSIGN) {
            ast_assign* assignStmt = (ast_assign*)stmt;
            ast_expr* rhs = assignStmt->get_rhs();
            //return apply(rhs);
        }
    }

    bool apply(ast_id* var) {
        
        if (var->get_parent()->get_nodetype() == AST_FIELD &&
            var == ((ast_field*)var->get_parent())->get_second()) {
            ast_field* parentField = (ast_field*)var->get_parent();
            //if (var == parentField->get_second()) {
                ast_id* firstField = parentField->get_first();
                
                symbol*s = transfer.findSymbol(firstField->get_orgname());
                if (!insideCudaKernel && s != NULL && !printingMacro) {
                    for (std::list<symbol*>::iterator it = s->children.begin(); it !=  s->children.end(); it++) {
                        if (strcmp((*it)->getName(), var->get_orgname()) == 0) {
                            printf("Field In expr = %s\n", var->get_orgname());
                            deviceFields.push_back(parentField);
                        }
                    }
                }
            //}
        } else {
            symbol*s = transfer.findSymbol(var->get_orgname());
            if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
                && !printingMacro) {
                printf("Var In expr = %s\n", var->get_orgname());
                deviceVariables.push_back(var);
            }
        }
        return true;
    }
};

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

    sprintf(temp, "%s/%s_main.cu", dname, fname);
    f_body = fopen(temp, "w");
    if (f_body == NULL) {
        gm_backend_error(GM_ERROR_FILEWRITE_ERROR, temp);
        return false;
    }
    Body.set_output_file(f_body);
    //get_lib()->set_code_writer(&Body);

    sprintf(temp, "%s/%s.cu", dname, fname);
    f_cudaBody = fopen(temp, "w");
    if (f_cudaBody == NULL) {
        gm_backend_error(GM_ERROR_FILEWRITE_ERROR, temp);
        return false;
    }
    cudaBody.set_output_file(f_cudaBody);
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
    if (f_cudaBody != NULL) {
        cudaBody.flush();
        fclose(f_cudaBody);
        if(remove_files) {
            sprintf(temp, "rm %s/%s.cu", dname, fname);
            system(temp);
        }
        f_cudaBody = NULL;
    }
}

void gm_cuda_gen::generate_lhs_id(ast_id* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    currentScope->addVariableToScope(i);
    //if (isOnGPUMemory(i)) {
    symbol*s = transfer.findSymbol(i->get_orgname());
    if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
        && !printingMacro) {
        Body.push("h_");
    }
    Body.push(i->get_orgname());
}

void gm_cuda_gen::generate_lhs_field(ast_field* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    Body.push(i->get_second()->get_orgname());
    symbol* iterator = transfer.findSymbol(i->get_first()->get_orgname());
    //std::string str = "[";
    //str = str + iterator->getName() + "]";
    //Body.push(str.c_str());
    Body.push("[");
    generate_lhs_id(i->get_first());
    Body.push("]");
    currentScope->addVariableToScope(i->get_first());
}

void gm_cuda_gen::generate_rhs_id(ast_id* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    currentScope->addVariableToScope(i);
    symbol*s = transfer.findSymbol(i->get_orgname());
    if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
        && !printingMacro) {
        Body.push("h_");
    }
    Body.push(i->get_orgname());
}

void gm_cuda_gen::generate_rhs_field(ast_field* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    Body.push(i->get_second()->get_orgname());
    symbol* iterator = transfer.findSymbol(i->get_first()->get_orgname());
    //std::string str = "[";
    //str = str + iterator->getName() + "]";
    //Body.push(str.c_str());
    Body.push("[");
    generate_rhs_id(i->get_first());
    Body.push("]");
    currentScope->addVariableToScope(i->get_first());
}
/*
void gm_cuda_gen::generate_expr_foreign(ast_expr* e) {

}*/

// TODO: Today
// builtin functions like G.NumEdges()
void gm_cuda_gen::generate_expr_builtin(ast_expr* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    ast_expr_builtin* b = (ast_expr_builtin*)i;
    gm_builtin_def* def = b->get_builtin_def();
    std::string str, graphName;
    symbol* iterSym;
    std::list<ast_expr*>::iterator itStart, itEnd;
    std::list<ast_argdecl*> inArgs;
    std::list<ast_argdecl*>::iterator It;
    ast_typedecl* type;
    ast_idlist* idlist;
    ast_id* id;
    switch (def->get_method_id()) {
        case GM_BLTIN_GRAPH_NUM_NODES: 
                Body.push("NumNodes");
                break;  
        case GM_BLTIN_GRAPH_NUM_EDGES: 
                Body.push("NumEdges");
                break;  
        case GM_BLTIN_GRAPH_RAND_NODE: 
                Body.push("BUILTIN3");
                break;  

        case GM_BLTIN_NODE_DEGREE: 
                str = "G0[" + std::string(b->get_driver()->get_orgname()) + " + 1] - G0[" + b->get_driver()->get_orgname() + "]";
                Body.push(str.c_str());
                break;      
        case GM_BLTIN_NODE_IN_DEGREE: 
                str = "G0[" + std::string(b->get_driver()->get_orgname()) + " + 1] - G0[" + b->get_driver()->get_orgname() + "]";
                Body.push(str.c_str());
                break;   
        case GM_BLTIN_NODE_TO_EDGE: 
                //Body.push("BUILTIN6");
                //str = std::string(b->get_driver()->get_orgname());
                //iterSym = transfer.findSymbol(b->get_driver()->get_orgname());
                //Body.push(str.c_str());
                Body.push("iter");
                break;     
        case GM_BLTIN_NODE_IS_NBR_FROM: 
                Body.push("BUILTIN7");
                break; 
        case GM_BLTIN_NODE_HAS_EDGE_TO: 
                Body.push("hasEdgeTo(");
                inArgs = currentProc->get_in_args();
                for (It = inArgs.begin(); It != inArgs.end(); It++) {
                    type = (*It)->get_type();
                    idlist = (*It)->get_idlist();
                    for (int ii = 0; ii < idlist->get_length(); ii++) {
                        id = idlist->get_item(ii);
                        if (type->is_graph()) {
                            graphName = std::string(id->get_orgname());
                            break;
                        }
                    }
                }
                str = graphName + "0, " + graphName + "1, ";
                Body.push(str.c_str());
                str = std::string(b->get_driver()->get_orgname()) + ", ";
                //iterSym = transfer.findSymbol(b->get_driver()->get_orgname());
                Body.push(str.c_str());
                itStart = b->get_args().begin();
                itEnd = b->get_args().end();
                for (; itStart != itEnd; itStart++) {
                    //(*itStart)->dump_tree(4);
                    generate_expr(*itStart);
                }
                Body.push(")");
                addDeviceFunction(GM_BLTIN_NODE_HAS_EDGE_TO);
                break; 
        case GM_BLTIN_NODE_RAND_NBR: 
                Body.push("BUILTIN9");
                break;    

        case GM_BLTIN_EDGE_FROM: 
                Body.push("BUILTIN10");
                break;        
        case GM_BLTIN_EDGE_TO: 
                Body.push("BUILTIN11");
                break;          

        case GM_BLTIN_TOP_DRAND: 
                Body.push("BUILTIN12");
                break;        
        case GM_BLTIN_TOP_IRAND: 
                Body.push("BUILTIN13");
                break;        
        case GM_BLTIN_TOP_LOG: 
                Body.push("BUILTIN14");
                break;          
        case GM_BLTIN_TOP_EXP: 
                Body.push("BUILTIN15");
                break;          
        case GM_BLTIN_TOP_POW: 
                Body.push("BUILTIN16");
                break;          

        case GM_BLTIN_SET_ADD: 
                Body.push("BUILTIN17");
                break;
        case GM_BLTIN_SET_REMOVE: 
                Body.push("BUILTIN18");
                break;
        case GM_BLTIN_SET_HAS: 
                Body.push("BUILTIN19");
                break;
        case GM_BLTIN_SET_ADD_BACK: 
                Body.push("BUILTIN20");
                break;
        case GM_BLTIN_SET_REMOVE_BACK: 
                Body.push("BUILTIN21");
                break;
        case GM_BLTIN_SET_PEEK: 
                Body.push("BUILTIN22");
                break;
        case GM_BLTIN_SET_PEEK_BACK: 
                Body.push("BUILTIN23");
                break;
        case GM_BLTIN_SET_UNION: 
                Body.push("BUILTIN24");
                break;
        case GM_BLTIN_SET_INTERSECT: 
                Body.push("BUILTIN25");
                break;
        case GM_BLTIN_SET_COMPLEMENT: 
                Body.push("BUILTIN26");
                break;
        case GM_BLTIN_SET_SUBSET: 
                Body.push("BUILTIN27");
                break;
        case GM_BLTIN_SET_SIZE: 
                Body.push("BUILTIN28");
                break;
        case GM_BLTIN_SET_CLEAR: 
                Body.push("BUILTIN29");
                break;

        case GM_BLTIN_SEQ_POP_FRONT: 
                Body.push("BUILTIN30");
                break;

        case GM_BLTIN_MAP_SIZE: 
                Body.push("BUILTIN31");
                break;         
        case GM_BLTIN_MAP_HAS_MAX_VALUE: 
                Body.push("BUILTIN32");
                break;
        case GM_BLTIN_MAP_HAS_MIN_VALUE: 
                Body.push("BUILTIN33");
                break;
        case GM_BLTIN_MAP_HAS_KEY: 
                Body.push("BUILTIN34");
                break;      
        case GM_BLTIN_MAP_GET_MAX_KEY: 
                Body.push("BUILTIN35");
                break;  
        case GM_BLTIN_MAP_GET_MIN_KEY: 
                Body.push("BUILTIN36");
                break;  
        case GM_BLTIN_MAP_GET_MAX_VALUE: 
                Body.push("BUILTIN37");
                break;
        case GM_BLTIN_MAP_GET_MIN_VALUE: 
                Body.push("BUILTIN38");
                break;
        case GM_BLTIN_MAP_CLEAR: 
                Body.push("BUILTIN39");
                break;        
        case GM_BLTIN_MAP_REMOVE: 
                Body.push("BUILTIN40");
                break;       

        case GM_BLTIN_MAP_REMOVE_MIN: 
                Body.push("BUILTIN41");
                break;   
        case GM_BLTIN_MAP_REMOVE_MAX: 
                Body.push("BUILTIN42");
                break;   
    }
}

void gm_cuda_gen::generate_expr_minmax(ast_expr* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
}

void gm_cuda_gen::generate_expr_abs(ast_expr* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
}

void gm_cuda_gen::generate_expr_nil(ast_expr* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
    Body.push("-1");
}

/*void gm_cuda_gen::generate_expr_type_conversion(ast_expr* e) {

}*/

const char* gm_cuda_gen::get_type_string(ast_typedecl* t) {

    if ((t == NULL) || (t->is_void())) {
        return "void";
    }

    if (t->is_primitive()) {
        return get_type_string(t->get_typeid());
    } else if (t->is_nodeedge()) {
        return get_type_string(t->get_typeid());
    } else if (t->is_property()) {
        ast_typedecl* t2 = t->get_target_type();
        assert(t2 != NULL);
        if (t2->is_primitive()) {
            switch (t2->get_typeid()) {
                case GMTYPE_BYTE:
                    return "char_t*";
                case GMTYPE_SHORT:
                    return "int16_t*";
                case GMTYPE_INT:
                    return "int32_t*";
                case GMTYPE_LONG:
                    return "int64_t*";
                case GMTYPE_FLOAT:
                    return "float*";
                case GMTYPE_DOUBLE:
                    return "double*";
                case GMTYPE_BOOL:
                    return "bool*";
                default:
                    assert(false);
                    break;
            }
        } else if (t2->is_nodeedge()) {
            char temp[128];
            sprintf(temp, "%s*", /*get_lib()->*/get_type_string(t2));
            return gm_strdup(temp);
        } else if (t2->is_collection()) {
            char temp[128];
            sprintf(temp, "%s<%s>&", "prop"/*PROP_OF_COL*/, /*get_lib()->*/get_type_string(t2));
            return gm_strdup(temp);
        } else {
            assert(false);
        }
    } else if (t->is_map()) {
        char temp[256];
        ast_maptypedecl* mapType = (ast_maptypedecl*) t;
        const char* keyType = get_type_string(mapType->get_key_type());
        const char* valueType = get_type_string(mapType->get_value_type());
        sprintf(temp, "gm_map<%s, %s>", keyType, valueType);
        return gm_strdup(temp);
    } else if (t->is_graph()) {
        return "int* ";
    } else
        assert(false);
        //return /*get_lib()->*/get_type_string(t);

    return "ERROR";
}

const char* gm_cuda_gen::get_type_string(int type_id) {
    
    if (gm_is_prim_type(type_id)) {
        switch (type_id) {
            case GMTYPE_BYTE:
                return "int8_t";
            case GMTYPE_SHORT:
                return "int16_t";
            case GMTYPE_INT:
                //return "int32_t";
                return "int";
            case GMTYPE_LONG:
                return "unsigned long long int";
            case GMTYPE_FLOAT:
                return "float";
            case GMTYPE_DOUBLE:
                return "double";
            case GMTYPE_BOOL:
                return "bool";
            default:
                assert(false);
                return "??";
        }
    } else if (gm_is_nodeedge_type(type_id)){
        return "int";
    } else {
        return "UnknownType";//get_lib()->get_type_string(type_id);
    }
}

/*
void gm_cuda_gen::generate_expr_list(std::list<ast_expr*>& L) {
}

void gm_cuda_gen::generate_expr(ast_expr* e) {
}

void gm_cuda_gen::generate_expr_val(ast_expr* e) {
}*/

void gm_cuda_gen::generate_expr_inf(ast_expr* e) {

    char* temp = temp_str;
    assert(e->get_opclass() == GMEXPR_INF);
    int t = e->get_type_summary();
    switch (t) {
        case GMTYPE_INF:
        case GMTYPE_INF_INT:
            sprintf(temp, "%s", e->is_plus_inf() ? "2147483647"/*"99999"*/ : "0"); // temporary
            break;
        case GMTYPE_INF_LONG:
            sprintf(temp, "%s", e->is_plus_inf() ? "LLONG_MAX" : "LLONG_MIN"); // temporary
            break;
        case GMTYPE_INF_FLOAT:
            sprintf(temp, "%s", e->is_plus_inf() ? "FLT_MAX" : "FLT_MIN"); // temporary
            break;
        case GMTYPE_INF_DOUBLE:
            sprintf(temp, "%s", e->is_plus_inf() ? "DBL_MAX" : "DBL_MIN"); // temporary
            break;
        default:
            sprintf(temp, "%s", e->is_plus_inf() ? "99999" : "0"); // temporary
            break;
    }
    _Body.push(temp);
    return;
}
/*
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

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");
}

std::string getAtomicUpdateInst(int reduceType) {

    switch (reduceType) {
        case GMREDUCE_PLUS: return std::string("atomicAdd");  break;
        // TODO: Use Atomic CAS
        case GMREDUCE_MULT:
        case GMREDUCE_MIN: return std::string("atomicMin");  break;
        case GMREDUCE_MAX: return std::string("atomicMax");  break;
        case GMREDUCE_AND: return std::string("atomicAnd");  break;
        case GMREDUCE_OR:  return std::string("atomicOr");  break;
        // TODO:
        case GMREDUCE_AVG:      break;
        case GMREDUCE_DEFER:    break;
    }
}

std::string gm_cuda_gen::getNewTempVariable(ast_node* n) {

    std::string str;
    if (n->get_nodetype() == AST_ID) {
        str = std::string("local") + ((ast_id*)n)->get_orgname();
    } else if (n->get_nodetype() == AST_FIELD) {
        ast_field* f = (ast_field*)n;
        str = std::string("local") + f->get_first()->get_orgname();
        str = str + f->get_second()->get_orgname(); 
    }
    return str;
}

void gm_cuda_gen::generateMacroDefine(scope* s) {
    
    gm_code_writer *tempBody = new gm_code_writer();
    *tempBody = Body;
    Header.flush();
    Body = Header;
    std::list<ast_vardecl*>varDecls = s->getVarDecls();
    std::list<ast_vardecl*>::iterator varDeclIt = varDecls.begin();
    Body.push("#define ");
    Body.push(s->getMacroName().c_str());
    Body.pushln(" \\");
    //Body.push("{");
    printingMacro = true;
    for (; varDeclIt != varDecls.end(); varDeclIt++) {
        if (s == GPUMemoryScope)
            Body.push("__device__ ");
        generate_sent_vardecl(*varDeclIt);
        printf("New generation of decl.\n");
        (*varDeclIt)->dump_tree(2);
        printf("Ended Generation\n");
    }
    if (s == GPUMemoryScope) {
        std::string str = "__device__ bool* gm_threadBlockBarrierReached;";
        Body.push(str.c_str());
        Body.pushln("  \\");
    } else if (s == globalScope) {
        std::string str = "bool* host_threadBlockBarrierReached;";
        Body.push(str.c_str());
        Body.pushln("  \\");
        str = "cudaError_t err;";
        Body.push(str.c_str());
        Body.pushln("  \\");
        Body.pushln("cudaEvent_t start, stop;");
        Body.pushln("float elapsedTime;");
        Body.pushln("int gm_blockSize;   /* The launch configurator returned block size*/ \\");
        Body.pushln("int gm_minGridSize; /* The minimum grid size needed to achieve the*/ \\");
        Body.pushln("           /* maximum occupancy for a full device launch*/ \\");
        Body.pushln("int gm_gridSize;    /* The actual grid size needed, based on input size*/ \\");
        Body.pushln("int gm_numBlocksStillToProcess, gm_offsetIntoBlocks; \\");
        Body.pushln("int gm_numBlocksKernelParameter; \\");

    }
    printingMacro = false;
    Body.pushln(" ");
    
    if (s == globalScope) {
        Body.pushln("#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }");
        Body.pushln("inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {");
        Body.pushln("if (code != cudaSuccess) {");
        Body.pushln("fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);");
        Body.pushln("if (abort) exit(code);");
        Body.pushln("}");
        Body.pushln("}");
        Body.pushln("");
 
        Body.pushln("#define CUDA_ERR_CHECK  \\");
        Body.pushln("gpuErrchk(cudaPeekAtLastError());\\");
        Body.pushln("gpuErrchk(cudaDeviceSynchronize());\\");
        Body.pushln("");

/*        Body.pushln("if( err != cudaSuccess) { \\");
        Body.pushln("printf(\"CUDA error: %s ** at Line %d in file %s\\n\", cudaGetErrorString(err), __LINE__, __FILE__); \\");
        Body.pushln("    return EXIT_FAILURE; \\");
        Body.pushln("}\\");
        Body.pushln(" ");*/

        std::list<int> funcList = getDeviceFunctionList();
        std::string str, graphName;
        for (std::list<int>::iterator funcListIt = funcList.begin(); funcListIt != funcList.end(); funcListIt++) {
            switch(*funcListIt) {

                case GM_BLTIN_NODE_HAS_EDGE_TO: 
                    Body.pushln("__device__ bool hasEdgeTo(int* gm_G0, int* gm_G1, int gm_from, int gm_to) {");
                    
                    str = "for (int i = gm_G0[gm_from]; i < gm_G0[gm_from + 1]; i++) {";
                    Body.pushln(str.c_str());
                    str = "if (gm_G1[i] == gm_to) {";
                    Body.pushln(str.c_str());
                    Body.pushln("return true;");
                    Body.pushln("}");

                    Body.pushln("}");
                    Body.pushln("return false;");
                    Body.pushln("}");
                    break;
            }
        }
    }
    Body.flush();
    Body = *tempBody;
}

void gm_cuda_gen::generate_sent_reduce_assign(ast_assign* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");

    if (!i->has_lhs_list()) {

        Body.push(getAtomicUpdateInst(i->get_reduce_type()).c_str());
        Body.push("(");
        if (i->is_target_scalar()) {
            if(gm_is_boolean_type(i->get_lhs_scala()->getTypeSummary())) {
            
                Body.push("(int *)");
            }
        } else {
            if (gm_is_boolean_type(i->get_lhs_field()->getTargetTypeSummary())) {
                Body.push("(int)");
            }
        }
        Body.push("&");
        if (i->is_target_scalar()) {
            generate_lhs_id(i->get_lhs_scala());
        } else {
            generate_lhs_field(i->get_lhs_field());
        }
        Body.push(", ");
        if(gm_is_boolean_type(i->get_rhs()->get_type_summary())) {
        
            Body.push("(int)");
        }
        generate_expr(i->get_rhs());
        Body.pushln(");");
    } else {
        std::string str;
        std::list<ast_node*>::iterator IStart = i->get_lhs_list().begin();
        std::list<ast_node*>::iterator IEnd = i->get_lhs_list().end();
        bool isBooleanReduction = false;
        for (; IStart != IEnd; IStart++) {
            if ((*IStart)->get_nodetype() == AST_FIELD) {
                ast_field* targetField = (ast_field*)*IStart;
                if (gm_is_boolean_type(targetField->getTargetTypeSummary())) {
                    isBooleanReduction = true;
                }
            }
        }
        // For SSSP boolean multiple reduction
        if (isBooleanReduction) {
            Body.push("localExpr = ");
            generate_lhs_field(i->get_lhs_field());
            Body.pushln(";");
            ast_id* localExpr = ast_id::new_id("localExpr", 0, 1);
            currentScope->addVariableToScope(localExpr, i->get_rhs()->get_type_summary());
            
            Body.push("expr = ");
            ast_id* newId = ast_id::new_id("expr", 0, 1);
            // TODO: Get right symtab info
            ast_typedecl* newType = getNewTypeDecl(i->get_rhs()->get_type_summary());
            currentScope->addVariableToScope(newId, i->get_rhs()->get_type_summary());
            generate_expr(i->get_rhs());
            Body.pushln(";");

            Body.push(getAtomicUpdateInst(i->get_reduce_type()).c_str());
            Body.push("(&");
            if (i->is_target_scalar()) {
                generate_lhs_id(i->get_lhs_scala());
            } else {
                generate_lhs_field(i->get_lhs_field());
            }
            Body.push(", ");
            Body.push("expr");
            Body.pushln(");");

            
            Body.push("if (localExpr ");
            // TODO
            //str = getNewVariable(integer, "localExpr");
            if (i->get_reduce_type() == GMREDUCE_MAX) {
                Body.push("<= ");
                currentScope->setLocalExprValue(2);
            }
            else if (i->get_reduce_type() == GMREDUCE_MIN) {
                Body.push("> ");
                currentScope->setLocalExprValue(5);
            }
            else
                assert(false);

            Body.pushln("expr) {");

            std::list<ast_node*>::iterator lhsListIt = i->get_lhs_list().begin();
            std::list<ast_expr*>::iterator rhsListIt = i->get_rhs_list().begin();
            std::list<ast_node*>::iterator lhsListEnd = i->get_lhs_list().end();
            std::list<ast_expr*>::iterator rhsListEnd = i->get_rhs_list().end();
            for (; (lhsListIt != lhsListEnd) && 
                    (rhsListIt != rhsListEnd); 
                    lhsListIt++, rhsListIt++) {
                generate_lhs_field((ast_field*)*lhsListIt);
                Body.push(" = ");
                generate_expr(*rhsListIt);
                Body.pushln(";");
            }
            Body.pushln("}");
            return;
        }
        // TODO
        //str = getNewVariable(integer, "expr");
        Body.push("expr = ");
        ast_id* newId = ast_id::new_id("expr", 0, 1);
        // TODO: Get right symtab info
        ast_typedecl* newType = getNewTypeDecl(i->get_rhs()->get_type_summary());
        currentScope->addVariableToScope(newId, i->get_rhs()->get_type_summary());
        generate_expr(i->get_rhs());
        Body.pushln(";");

        Body.push(getAtomicUpdateInst(i->get_reduce_type()).c_str());
        Body.push("(&");
        if (i->is_target_scalar()) {
            generate_lhs_id(i->get_lhs_scala());
        } else {
            generate_lhs_field(i->get_lhs_field());
        }
        Body.push(", ");
        Body.push("expr");
        //generate_expr(i->get_rhs());
        Body.pushln(");");

        /*printf("Reduce ASSIGN######\n");
        std::list<ast_node*> lhsList = i->get_lhs_list();
        printf("LHS List Nodes:");
        for (std::list<ast_node*>::iterator it = lhsList.begin();
             it != lhsList.end(); it++) {
            ast_node* lhsVar = *it;
            lhsVar->dump_tree(2);
        }
        std::list<ast_expr*> rhsList = i->get_rhs_list();
        if (rhsList.size() > 0) {
            printf("Has rhs list\n");
            printf("RHS List Nodes:");
            for (std::list<ast_expr*>::iterator it = rhsList.begin();
                 it != rhsList.end(); it++) {
                ast_expr* rhsVar = *it;
                rhsVar->dump_tree(2);
            }
        }
        printf("\nBound Start\n");
        i->get_bound()->dump_tree(2);
        printf("Bound Stop\n");
        printf("arg_minmax = %d\n", i->is_argminmax_assign());
        printf("is_reference = %d\n", i->is_reference());
        printf("\n");*/

        assert(i->get_reduce_type() == GMREDUCE_MAX ||
               i->get_reduce_type() == GMREDUCE_MIN);

        newId = ast_id::new_id("localExpr", 0, 1);
        currentScope->addVariableToScope(newId, i->get_rhs()->get_type_summary());
        Body.push("if (localExpr ");
        // TODO
        //str = getNewVariable(integer, "localExpr");
        if (i->get_reduce_type() == GMREDUCE_MAX) {
            Body.push("<= ");
            currentScope->setLocalExprValue(2);
        }
        else if (i->get_reduce_type() == GMREDUCE_MIN) {
            Body.push(">= ");
            currentScope->setLocalExprValue(5);
        }
        else
            assert(false);

        Body.pushln("expr) {");
        Body.pushln("localExpr = expr;");

        ast_sentblock* newStmts = ast_sentblock::new_sentblock();
        
        ast_id* barrierId = ast_id::new_id("softwareBarrier", 0, 1);
        ast_call* barrierCall = ast_call::new_call(barrierId);
        newStmts->add_sent(barrierCall);
        
        newId = ast_id::new_id("localExpr", 0, 1);
        ast_expr* condLeft = ast_expr::new_id_expr(newId);
        ast_expr* condRight;
        if (i->is_target_scalar()) {
            condRight = ast_expr::new_id_expr(i->get_lhs_scala());
        } else {
            ast_field* targetField = i->get_lhs_field();
            std::string str = std::string("local") + targetField->get_first()->get_orgname();
            ast_id* localVar = ast_id::new_id(str.c_str(), 0, 1);
            ast_field* localField = ast_field::new_field(localVar, targetField->get_second());
            condRight = ast_expr::new_field_expr(localField);
        }
        ast_expr* ifCond = ast_expr::new_biop_expr(GMOP_EQ, condLeft, condRight);

        ast_id* chooseThreadVar = ast_id::new_id("chooseThread", 0, 1);
        //symbol* newSymbol = transfer.addToSymTableAndReturn(chooseThreadVar);
        //newSymbol->setMemLoc(GPUMemory);
        GPUMemoryScope->addVariableToScope(chooseThreadVar, GMTYPE_INT);
        ast_expr* threadId = ast_expr::new_id_expr(ast_id::new_id("tId", 0, 1));
        ast_sent* chooseThreadAssign = ast_assign::new_assign_scala(chooseThreadVar, threadId);
        ast_sent* newIf = ast_if::new_if(ifCond, chooseThreadAssign, NULL);
        newStmts->add_sent(newIf);
        newStmts->add_sent(barrierCall);

        std::list<ast_node*>::iterator lhsListIt = i->get_lhs_list().begin();
        std::list<ast_expr*>::iterator rhsListIt = i->get_rhs_list().begin();
        std::list<ast_node*>::iterator lhsListEnd = i->get_lhs_list().end();
        std::list<ast_expr*>::iterator rhsListEnd = i->get_rhs_list().end();
        ast_sentblock* assignToGlobals = ast_sentblock::new_sentblock();
        for (; (lhsListIt != lhsListEnd) && 
                (rhsListIt != rhsListEnd); 
                lhsListIt++, rhsListIt++) {
            Body.push(getNewTempVariable(*lhsListIt).c_str());
            // TODO
            //str = getNewVariable(integer, "local" + id->get_orgname(),);
            Body.push(" = ");
            generate_expr(*rhsListIt);
            Body.pushln(";");

            if ((*lhsListIt)->get_nodetype() == AST_ID) {
                ast_id* targetId = (ast_id*)*lhsListIt;
                std::string str = getNewTempVariable(targetId);
                ast_id* localVar = ast_id::new_id(str.c_str(), 0, 1);
                ast_expr* localExpr = ast_expr::new_id_expr(localVar);
                ast_assign* newAssign = ast_assign::new_assign_scala(targetId, localExpr);
                assignToGlobals->add_sent(newAssign);
                globalScope->addVariableToScope(targetId, targetId->getTypeSummary());
                currentScope->addVariableToScope(localVar, targetId->getTypeSummary());
            } else if ((*lhsListIt)->get_nodetype() == AST_FIELD) {
                ast_field* targetField = (ast_field*)*lhsListIt;
                std::string str = std::string("local") + targetField->get_first()->get_orgname();
                ast_id* localVar = ast_id::new_id(str.c_str(), 0, 1);
                ast_field* localField = ast_field::new_field(localVar, targetField->get_second());
                //transfer.addFieldToSymTable(localField);
                int iterType = getNewTypeDecl(targetField->getTypeSummary())->get_typeid();
                currentScope->addVariableToScope(localVar, iterType);
                str = getNewTempVariable(targetField);
                localVar = ast_id::new_id(str.c_str(), 0, 1);
                ast_expr* localExpr = ast_expr::new_id_expr(localVar);
                ast_assign* newAssign = ast_assign::new_assign_field(localField, localExpr);
                assignToGlobals->add_sent(newAssign);
                currentScope->addVariableToScope(localVar, targetField->getTargetTypeSummary());

                str = getNewTempVariable(targetField->get_first());
                Body.push(str.c_str());
                Body.push(" = ");
                Body.push(targetField->get_first()->get_orgname());
                Body.pushln(";");
            }
        }
        condLeft = ast_expr::new_id_expr(chooseThreadVar);
        condRight = threadId;
        ifCond = ast_expr::new_biop_expr(GMOP_EQ, condLeft, condRight);
        newIf = ast_if::new_if(ifCond, assignToGlobals, NULL);
        newStmts->add_sent(newIf);

        ast_id* boundIter = i->get_bound();
        ast_node* parent = i->get_parent();
        ast_foreach* forEachStmt;
        bool found = false;
        while(!found) {
            if (parent->get_nodetype() == AST_FOREACH) {
                forEachStmt = (ast_foreach*)parent;
                if (forEachStmt->is_parallel() &&
                    !strcmp(forEachStmt->get_iterator()->get_orgname(), boundIter->get_orgname())) {
                    printf("Found the bounding foreach\n");
                    found = true;
                }
            }
            parent = parent->get_parent();
        }
        forEachStmt->set_GlobalBarrier(true);
        forEachStmt->setNewStmtsAfterBarrier(newStmts);
        Body.pushln("}");
    }
}

/*void gm_cuda_gen::generate_sent_defer_assign(ast_assign* i) {

}*/

void gm_cuda_gen::generate_sent_vardecl(ast_vardecl* v) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    v->dump_tree(2);
    printf("\nEnded..\n\n");
    ast_typedecl* t = v->get_type();

    if (t->is_collection_of_collection()) {
        Body.push(get_type_string(t));
        ast_typedecl* targetType = t->get_target_type();
        Body.push("<");
        Body.push(get_type_string(t->getTargetTypeSummary()));
        Body.push("> ");
        ast_idlist* idl = v->get_idlist();
        assert(idl->get_length() == 1);
        generate_lhs_id(idl->get_item(0));
        //get_lib()->add_collection_def(idl->get_item(0));
        return;
    }

    if (t->is_map()) {
        ast_maptypedecl* map = (ast_maptypedecl*) t;
        ast_idlist* idl = v->get_idlist();
        assert(idl->get_length() == 1);
        //get_lib()->add_map_def(map, idl->get_item(0));
        return;
    }

    if (t->is_sequence_collection()) {
        //for sequence-list-vector optimization
        ast_id* id = v->get_idlist()->get_item(0);
        const char* type_string;
        /*if(get_lib()->has_optimized_type_name(id->getSymInfo())) {
            type_string = get_lib()->get_optimized_type_name(id->getSymInfo());
        } else*/ {
            type_string = get_type_string(t);
        }
        Body.push_spc(type_string);
    } else if (t->is_graph()){
        Body.push_spc("int* ");
    } else {
        Body.push_spc(get_type_string(t));
    }

    if (t->is_property()) {
        ast_idlist* idl = v->get_idlist();
        //assert(idl->get_length() == 1);
        //generate_lhs_id(idl->get_item(0));
        //declare_prop_def(t, idl->get_item(0));
        generate_idlist_primitive(idl);
        Body.push(";");
        if (printingMacro)
            Body.pushln("   \\");
        else
            Body.pushln("");
    } else if (t->is_collection()) {
        ast_idlist* idl = v->get_idlist();
        //assert(idl->get_length() == 1);
        generate_lhs_id(idl->get_item(0));
        //get_lib()->add_collection_def(idl->get_item(0));
    } else if (t->is_primitive()) {
        generate_idlist_primitive(v->get_idlist());
        Body.push(";");
        if (printingMacro)
            Body.pushln("   \\");
        else
            Body.pushln("");
    } else {
        generate_idlist(v->get_idlist());
        Body.push(";");
        if (printingMacro)
            Body.pushln("   \\");
        else
            Body.pushln("");
    }
}

void gm_cuda_gen::generate_sent_foreach(ast_foreach* i) {

    printf("Entered Line number %d\n", __LINE__);
    if(doPrint)
    i->dump_tree(2);
    printf("\nEnded..\n\n");

    //Body.push("}\n");

    gm_code_writer *tempBody = new gm_code_writer();
    std::string callStr;
    if (i->is_parallel()) {
        insideCudaKernel = true;
        //Body.flush();
        *tempBody = Body;
        Body = cudaBody;
        scope* newScope = new scope(i, currentProc);
        newScope->setParentScope(getCurrentScope());
        newScope->setGPUScope(getGPUScope());
        setCurrentScope(newScope);
        addToKernelScopes(newScope);

        callStr = generate_newKernelFunction(i);
    }

    generate_CudaAssignForIterator(i->get_iterator(), i->is_parallel());

    if (i->get_filter() != NULL) {
        printf("For Each Conditional %%%%%%%%%%\n");
    }

    generate_sent(i->get_body());

    if (i->is_parallel()) {
        // Create Global Barrier after the foreach parallel loop
        if (i->need_GlobalBarrier()) {
            assert(i->is_parallel() && i->need_GlobalBarrier());
            generate_sent_block(i->getNewStmtsAfterBarrier(), false);
        }
         
        Body.pushln("}");
        Body.flush();

        insideCudaKernel = false;
        Body = *tempBody;
        Body.push(callStr.c_str());

        Body.pushln("CUDA_ERR_CHECK;");
        Body.pushln("gm_numBlocksStillToProcess -= gm_minGridSize;"); 
        Body.pushln("gm_offsetIntoBlocks += gm_minGridSize * gm_blockSize;");
        Body.pushln("}");
        std::list<scope_entry*>varList = currentScope->getScopeEntries();
        std::list<scope_entry*>::iterator varListIt = varList.begin();
        for (; varListIt != varList.end(); varListIt++) {
            ast_id* varId = (*varListIt)->getId();
            ast_typedecl* varType = (*varListIt)->getTypeDecl();
            if ((*varListIt)->getMemLoc() == GLOBAL_GPU_MEMORY)
                GPUMemoryScope->addVariableDeclaration(varId, varType);
            else {
                std::list<ast_argdecl*>& inArgs = currentProc->get_in_args();
                std::list<ast_argdecl*>::iterator i;
                
                bool found = false;
                for (i = inArgs.begin(); i != inArgs.end(); i++) {
                    ast_typedecl* type = (*i)->get_type();
                    ast_idlist* idlist = (*i)->get_idlist();
                    for (int ii = 0; ii < idlist->get_length(); ii++) {
                        ast_id* id = idlist->get_item(ii);
                        if (!strcmp(id->get_orgname(), varId->get_orgname())){
                            // Variable found as a argument
                            found = true;
                            break;
                        }
                    }
                    if(found)
                        break;
                }
                if (!found)
                    currentScope->addVariableDeclaration(varId, varType);
            }
        }

        generateMacroDefine(getCurrentScope());
        getCurrentScope()->printScopeVariables();
        setCurrentScope(getCurrentScope()->getParentScope());
    }
}

void gm_cuda_gen::generate_CudaAssignForIterator(ast_id* iter, bool isParallel) {

    symbol* iterSym = transfer.findSymbol(iter->get_orgname());
    if (isParallel) {
        Body.pushln("int tId = blockIdx.x * blockDim.x + threadIdx.x + gm_offsetIntoBlocks;");
        iterSym->setSymbolParallelIterator(true);
        iterSym->getIndexParallel();
        std::string str = "if (tId >= " + iterSym->getNumOfThreads() + ") {";
        Body.pushln(str.c_str());
        str = "return;";
        Body.pushln(str.c_str());
        Body.pushln("}");
        str = /*iterSym->getIteratorTypeString() + " " +*/ iterSym->getName();
        str = str + " = ";
        str = str + iterSym->getCodeGenName() + ";\n";
        Body.push(str.c_str());
    } else {
        iterSym->setSymbolParallelIterator(false);
        iterSym->getIndexSequential();
        std::string str = std::string("for (") + iterSym->getIteratorTypeString() + " iter" + /*iterSym->getName() +*/ " = " + iterSym->startIndexStr + ", ";
        str += iterSym->getName() + std::string(" = ");
        if (gm_is_any_neighbor_node_iteration(iterSym->getSymbolIterType()))
            str += "G1[iter]; ";
        else if (gm_is_any_neighbor_edge_iteration(iterSym->getSymbolIterType()))
            str += "G1[iter]; ";
        str += std::string("iter") + /*iterSym->getName() +*/ " < " + iterSym->endIndexStr + "; iter" + /*iterSym->getName() +*/ "++, ";
        
        str += iterSym->getName() + std::string(" = ");
        if (gm_is_any_neighbor_node_iteration(iterSym->getSymbolIterType()))
            str += "G1[iter]";
        else if (gm_is_any_neighbor_edge_iteration(iterSym->getSymbolIterType()))
            str += "G1[iter]";
        str += ") ";

        Body.push(str.c_str());
    }
}

std::string gm_cuda_gen::generate_newKernelFunction(ast_foreach* f) {

    static int forLoopId = 0;
    char temp[1024];
    sprintf(temp, "%d", forLoopId);
    std::string str = "__global__ void forEachKernel" + std::string(temp) + std::string(" (");
    std::string callStr;
    symbol* iterSym = transfer.findSymbol(f->get_iterator()->get_orgname());

    callStr = "cudaOccupancyMaxPotentialBlockSize(&gm_minGridSize, &gm_blockSize,";
    callStr += "forEachKernel" + std::string(temp) + ", 0, 0);\n";
    // Round up according to array size
    callStr += "gm_gridSize = (" + iterSym->getNumOfThreads() + " + gm_blockSize - 1) / gm_blockSize;\n";

    callStr += "gm_numBlocksStillToProcess = gm_gridSize, gm_offsetIntoBlocks = 0;\n";
    callStr += "while (gm_numBlocksStillToProcess > 0) {\n";
    callStr += "if (gm_numBlocksStillToProcess > gm_minGridSize)\n";
    callStr += "    gm_numBlocksKernelParameter = gm_minGridSize;\n";
    callStr += "else\n";
    callStr += "    gm_numBlocksKernelParameter = gm_numBlocksStillToProcess;\n";
 
    callStr += "forEachKernel" + std::string(temp) + std::string("<<<");
    callStr += "gm_numBlocksKernelParameter, gm_blockSize>>>(";

    Body.indent = 0;
    Body.push(str.c_str());

    std::list<ast_argdecl*>& inArgs = currentProc->get_in_args();
    std::list<ast_argdecl*>::iterator i;
    std::list<std::string>argList;
    bool isFirst = true;
    for (i = inArgs.begin(); i != inArgs.end(); i++) {
        ast_typedecl* type = (*i)->get_type();
        ast_idlist* idlist = (*i)->get_idlist();
        for (int ii = 0; ii < idlist->get_length(); ii++) {
            if (isFirst == false) {
                Body.push(", ");
                callStr = callStr + ", ";
            }
            isFirst = false;

            ast_id* id = idlist->get_item(ii);
            globalScope->addVariableDeclaration(id, type);
            bool found = false;
            for (std::list<std::string>::iterator strIt = argList.begin();
                 strIt != argList.end(); strIt++) {
                if (!strcmp(id->get_orgname(), (*strIt).c_str())) {
                    found = true;
                    break;
                }
            }
            if (found)
                continue;
            argList.push_back(id->get_orgname());

            if (type->is_primitive()) {

                sprintf(temp, "%s %s", gm_get_type_string(type->get_typeid()), id->get_orgname());
                callStr.append(id->get_orgname());
            }else if (type->is_graph()) {
                sprintf(temp, "int *%s0, int *%s1", id->get_orgname(), id->get_orgname());
                Body.push(temp);
                Body.push(", int NumNodes, int NumEdges");
                callStr.append(id->get_orgname());
                callStr.append("0, ");
                callStr.append(id->get_orgname());
                callStr.append("1, NumNodes, NumEdges");
                continue;
            }else if (type->is_property()) {
                ast_typedecl* targetType = type->get_target_type();
                sprintf(temp, "%s * %s", get_type_string(targetType->get_typeid()), id->get_orgname());
                callStr.append(id->get_orgname());
            }else if (type->is_nodeedge()) {

                sprintf(temp, "int %s", id->get_orgname());
                callStr.append(id->get_orgname());
            }else if (type->is_collection()) {

                sprintf(temp, "%s %s", gm_get_type_string(type->get_typeid()), id->get_orgname());
                callStr.append(id->get_orgname());
            }
            Body.push(temp);
        }
    }

    // Identify the dependencies in the foreach loop
    mapForEachToVariables::iterator forLoopsVarListIter = transfer.forEachLoopVariables.find(f);
    if (forLoopsVarListIter !=  transfer.forEachLoopVariables.end()) {
        listOfVariables varList = forLoopsVarListIter->second;
        for (listOfVariables::iterator it = varList.begin(); it != varList.end(); it++) {
            ast_node* varUsed = *it;
            if (varUsed->get_nodetype() == AST_FIELD) {
                ast_field* f = (ast_field*)varUsed;
                //ast_typedecl* targetType = f->getTypeInfo()->get_target_type();
                ast_typedecl* targetType = f->getTypeInfo()->get_target_type();
                
                globalScope->addVariableDeclaration(f->get_second(), f->getTypeInfo());
                bool found = false;
                for (std::list<std::string>::iterator strIt = argList.begin();
                     strIt != argList.end(); strIt++) {
                    if (!strcmp(f->get_second()->get_orgname(), (*strIt).c_str())) {
                        found = true;
                        break;
                    }
                }
                if (found)
                    continue;
                argList.push_back(f->get_second()->get_orgname());

                Body.push(", ");
                callStr.append(", ");
                sprintf(temp, "%s * %s", get_type_string(targetType->get_typeid()), f->get_second()->get_orgname());
                Body.push(temp);
                callStr.append(f->get_second()->get_orgname());
            }
        }
    }
    Body.push(", int gm_offsetIntoBlocks");
    Body.push(") {\n");
    
    // Generate Macro for declaration of local variables for the kernel
    sprintf(temp, "%d", forLoopId);
    str = "kernelMacro" + std::string(temp); 
    forLoopId++;
    currentScope->setMacroName(str);
    str = str + ";";
    Body.pushln(str.c_str());

    callStr.append(", gm_offsetIntoBlocks");
    callStr.append(");\n");
    return callStr;
}

void gm_cuda_gen::generate_sent_bfs(ast_bfs* i) {

}
/*
void gm_cuda_gen::generate_sent(ast_sent* i) {

}*/
std::string gm_cuda_gen::getTempVariable(ast_typedecl* varType) {

    static std::list<ast_vardecl*> tempVars;
    static int count = 0;
    for (std::list<ast_vardecl*>::iterator I = tempVars.begin(); I != tempVars.end(); I++) {
        ast_typedecl* type = (*I)->get_type();
        ast_idlist* idList = (*I)->get_idlist();
        if (type->get_typeid() == varType->get_typeid()) {
            std::string str; //= get_type_string(varType);
            str = /*" " +*/ std::string(idList->get_item(0)->get_orgname());
            return str;
        }
    }
    char temp[100];
    sprintf(temp, "%d", count);
    count++;
    std::string name = std::string("tempVar") + std::string(temp);
    ast_vardecl* newVar = ast_vardecl::new_vardecl(varType, ast_id::new_id(name.c_str(), 0, 1));
    tempVars.push_back(newVar);
    std::string str = std::string(get_type_string(varType)) + " " + name;
    return str;
}

void gm_cuda_gen::generate_sent_assign(ast_assign* i) {

    findDeviceVariablesInExpr deviceVarAnalysis(insideCudaKernel, printingMacro);
    i->get_rhs()->traverse_pre(&deviceVarAnalysis);
    std::list<ast_id*> deviceVars = deviceVarAnalysis.getVars();
    for (std::list<ast_id*>::iterator I = deviceVars.begin(); I != deviceVars.end(); I++) {
        if ((*I)->getTypeInfo()->is_graph())
            continue;
        CUDAMemcpyFromSymbol(*I);
    }

    std::list<ast_field*> deviceFields = deviceVarAnalysis.getFields();
    for (std::list<ast_field*>::iterator I = deviceFields.begin(); I != deviceFields.end(); I++) {
        if ((*I)->getTypeInfo()->is_graph())
            continue;
        ast_field* f = *I;
        ast_id* id = f->get_first();
    std::string str = getTempVariable(f->getTargetTypeInfo());
    str += ";";
    Body.pushln(str.c_str());
    str = "err = cudaMemcpy(&" + getTempVariable(f->getTargetTypeInfo()) + ", ";
    str += std::string(f->get_second()->get_orgname()) + " + " + id->get_orgname() + ", ";
    str += getSizeOfVariable(f->get_first()) + " * " + std::string("sizeof(");
    str += get_type_string(f->get_second()->getTargetTypeInfo()) + std::string(")");
    str += ", cudaMemcpyDeviceToHost";
    str += ");";
    Body.pushln(str.c_str());
    Body.pushln("CUDA_ERR_CHECK;");
    }
    bool cond = false;

    if (i->is_target_scalar()) {
        ast_id* id = i->get_lhs_scala();
        symbol* s = transfer.findSymbol(id->get_orgname());
        if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
            && !printingMacro) {
            cond = true;
        }
    } else {
        ast_field* f = i->get_lhs_field();
        ast_id* id = f->get_first();
        symbol* s = transfer.findSymbol(id->get_orgname());
        if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
            && !printingMacro) {
            cond = true;
        }
    }

    if (!cond) {
        if (i->is_target_scalar()) {
            generate_lhs_id(i->get_lhs_scala());
        } else {
            generate_lhs_field(i->get_lhs_field());
        }
        Body.push(" = ");

        generate_expr(i->get_rhs());

        Body.pushln(";");
        return;
    }

    if (i->is_target_scalar()) {
        generate_lhs_id(i->get_lhs_scala());
        Body.push(" = ");
        generate_expr(i->get_rhs());
        Body.pushln(";");

        CUDAMemcpyToSymbol(i->get_lhs_scala());
    } else {
        ast_field* f = i->get_lhs_field();
        ast_id* id = f->get_first();
        std::string str = getTempVariable(f->getTargetTypeInfo());
        str += std::string(" = ");
        Body.push(str.c_str());
        generate_expr(i->get_rhs());
        Body.pushln(";");

        CUDAMemcpyFromSymbol(f->get_first());
        str = getTempVariable(f->getTargetTypeInfo());
        std::string typeString = getSizeOfVariable(f->get_first()) + " * " + std::string("sizeof(");
        typeString += get_type_string(f->get_second()->getTargetTypeInfo()) + std::string(")");
        CUDAMemcpy(f->get_second(), f->get_first(), str, typeString, true);
    }
}

void gm_cuda_gen::generate_sent_if(ast_if* i) {

    findDeviceVariablesInExpr deviceVarAnalysis(insideCudaKernel, printingMacro);
    i->get_cond()->traverse_pre(&deviceVarAnalysis);
    std::list<ast_id*> deviceVars = deviceVarAnalysis.getVars();
    for (std::list<ast_id*>::iterator I = deviceVars.begin(); I != deviceVars.end(); I++) {
        if ((*I)->getTypeInfo()->is_graph())
            continue;
        CUDAMemcpyFromSymbol(*I);
    }
    _Body.push("if (");
    generate_expr(i->get_cond());

    _Body.pushln(")");
    ast_sent *s = i->get_then();
    if (s->get_nodetype() != AST_SENTBLOCK) {
        _Body.push_indent();
    }

    generate_sent(s);

    if (s->get_nodetype() != AST_SENTBLOCK) {
        _Body.pop_indent();
    }

    s = i->get_else();
    if (s == NULL) return;

    _Body.push("else ");
    if (s->get_nodetype() != AST_IF) {
        _Body.NL();
    }
    if ((s->get_nodetype() != AST_SENTBLOCK) && (s->get_nodetype() != AST_IF)) {
        _Body.push_indent();
    }

    generate_sent(s);

    if ((s->get_nodetype() != AST_SENTBLOCK) && (s->get_nodetype() != AST_IF)) {
        _Body.pop_indent();
    }
}
/*
void gm_cuda_gen::generate_sent_while(ast_while* i) {

}
*/
/*
void gm_cuda_gen::generate_sent_block(ast_sentblock* i) {

}

void gm_cuda_gen::generate_sent_block(ast_sentblock* i, bool need_br) {

}*/

void gm_cuda_gen::generate_sent_return(ast_return* r) {

    findDeviceVariablesInExpr deviceVarAnalysis(insideCudaKernel, printingMacro);
    r->get_expr()->traverse_pre(&deviceVarAnalysis);
    std::list<ast_id*> deviceVars = deviceVarAnalysis.getVars();
    for (std::list<ast_id*>::iterator I = deviceVars.begin(); I != deviceVars.end(); I++) {
        if ((*I)->getTypeInfo()->is_graph())
            continue;
        CUDAMemcpyFromSymbol(*I);
    }
    Body.push("return");
    if (r->get_expr() != NULL) {
        Body.SPC();
        generate_expr(r->get_expr());
    }
    Body.pushln(";");
}

void gm_cuda_gen::generate_sent_call(ast_call* c) {

    //assert(c->is_builtin_call());
    if (c->is_builtin_call())
        generate_expr_builtin(c->get_builtin());
    else {
        //generate_lhs_id(c->getCallName());
        Body.push(c->getCallName()->get_orgname());
        Body.push("(");
        Body.push(")");
    }
    Body.pushln(";");
}
/*
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

void gm_cuda_gen::generate_idlist(ast_idlist* idl) {

    int z = idl->get_length();
    for (int i = 0; i < z; i++) {
        ast_id* id = idl->get_item(i);
        generate_lhs_id(id);
        if (id->getTypeInfo()->is_graph()) {
            Body.push_spc("0, *");
            generate_lhs_id(id);
            Body.push_spc("1");
            Body.push_spc(", NumNodes");
            Body.push_spc(", NumEdges");
        }
        if (i < z - 1) Body.push_spc(',');
    }
}

void gm_cuda_gen::generate_idlist_primitive(ast_idlist* idList) {
    int length = idList->get_length();
    for (int i = 0; i < length; i++) {
        ast_id* id = idList->get_item(i);
        generate_lhs_id(id);
        //generate_lhs_default(id->getTypeSummary());
        if (!strcmp(id->get_orgname(), "localExpr")) {
            int l = getCurrentScope()->getLocalExprValue();
            if (l == 2)
                Body.push_spc(" = 0");
            else if (l == 5)
                Body.push_spc(" = 999999");
        }
        if (i < length - 1) Body.push_spc(',');
        if ((i < length - 1) && id->getSymInfo() != NULL &&
            id->getTypeInfo()->is_property()) {
            if (printingMacro)
                Body.push("*");
        }
    }
}

void gm_cuda_gen::generate_lhs_default(int type) {
    switch (type) {
        case GMTYPE_BYTE:
        case GMTYPE_SHORT:
        case GMTYPE_INT:
        case GMTYPE_LONG:
            Body.push_spc(" = 0");
            break;
        case GMTYPE_FLOAT:
        case GMTYPE_DOUBLE:
            Body.push_spc(" = 0.0");
            break;
        case GMTYPE_BOOL:
            Body.push_spc(" = false");
            break;
        default:
            assert(false);
            return;
    }
}

void gm_cuda_gen::generate_proc(ast_procdef* proc) {

    printf("Inside Code Generation of CUDA\n");

    // Parallel Regions Analysis
    gm_cuda_par_regions PRA = transfer;
    currentProc = proc;
    scope* newScope = new scope(currentProc);

    std::string macroName = proc->get_procname()->get_orgname() + std::string("Macro");
    newScope->setMacroName(macroName);
    setGlobalScope(newScope);
    setCurrentScope(newScope);
    
    macroName = proc->get_procname()->get_orgname() + std::string("MacroGPU");
    newScope = new scope(currentProc);
    newScope->setMacroName(macroName);
    setGPUScope(newScope);
    
    getGlobalScope()->setGPUScope(newScope);
    newScope->setGPUScope(newScope);


    cudaBody.push(getGPUScope()->getMacroName().c_str());
    cudaBody.pushln(";");
    add_include("GlobalBarrier.cuh", cudaBody, false);
    cudaBody.flush();
    
    Body.push(getGlobalScope()->getMacroName().c_str());
    Body.pushln(";");
    char temp[1024];
    add_include("graph.h", Body, false);
    std::string str = "verify_" + std::string(proc->get_procname()->get_genname()) + ".h";
    add_include(str.c_str(), Body, false);
    generate_kernel_function(proc);

    generate_sent(proc->get_body());
    Body.pushln("}");

    markGPUAndCPUGlobal();

    generateMacroDefine(getCurrentScope());
    getCurrentScope()->printScopeVariables();
    printf("GlobalMemory Scope:\n");
    generateMacroDefine(getGPUScope());
    getGPUScope()->printScopeVariables();
    //setGlobalScope(NULL);
    setCurrentScope(NULL);
}

// Generate the Kernel call definition
void gm_cuda_gen::generate_kernel_function(ast_procdef* proc) {

    gm_code_writer& Out = Body;
    std::string str;
    ast_typedecl* ret_type = proc->get_return_type();
    str = get_type_string(ret_type) + std::string(" ");
    char temp[1024];

    str = str + proc->get_procname()->get_genname() + "_CPU";
    Out.push(str.c_str());

    Out.push("(");
    
    std::list<ast_argdecl*>& inArgs = proc->get_in_args();
    std::list<ast_argdecl*>::iterator i;
    bool isFirst = true;
    for (i = inArgs.begin(); i != inArgs.end(); i++) {
        ast_typedecl* type = (*i)->get_type();
        ast_idlist* idlist = (*i)->get_idlist();
        for (int ii = 0; ii < idlist->get_length(); ii++) {
            if (isFirst == false)
                Out.push(", ");
            isFirst = false;

            ast_id* id = idlist->get_item(ii);
            if (type->is_primitive()) {

                sprintf(temp, "%s %s", get_type_string(type->get_typeid()), id->get_orgname());
            }else if (type->is_graph()) {
                sprintf(temp, "int* %s0, int* %s1", id->get_orgname(), id->get_orgname());
            }else if (type->is_property()) {
                ast_typedecl* targetType = type->get_target_type();
                sprintf(temp, "%s * %s", get_type_string(targetType->get_typeid()), id->get_orgname());
            }else if (type->is_nodeedge()) {

                sprintf(temp, "int %s", id->get_orgname());
            }else if (type->is_collection()) {

                sprintf(temp, "%s %s", gm_get_type_string(type->get_typeid()), id->get_orgname());
            }
            Out.push(temp);
        }
    }
    //Out.push();

    Out.push(") {\n");

    Out.NL();

}

/*
void gm_cuda_gen::generate_mapaccess(ast_expr_mapaccess* e) {

} */

std::string gm_cuda_gen::getSizeOfVariable(ast_id* i) {

    int typeId = i->getTypeSummary();
    if (gm_is_prim_type(typeId))
        return std::string("1");
    else if (gm_is_node_property_type(typeId))
        return std::string("NumNodes");
    else if (gm_is_edge_property_type(typeId))
        return std::string("NumEdges");
    else if (gm_is_nodeedge_type(typeId))
        return std::string("1");
    else
        return std::string("INVALID");
}

void gm_cuda_gen::CUDAAllocateMemory(ast_vardecl* varDecl, bool onHost) {

    ast_typedecl* varType = varDecl->get_type();
    ast_idlist* idList = varDecl->get_idlist();
    std::string str, str2, errCheck = "CUDA_ERR_CHECK;";
    for (int i = 0; i < idList->get_length(); i++) {
        ast_id* varId = idList->get_item(i);
        if (varId->getSymInfo() == NULL)
            continue;
        int typeId = varId->getTypeSummary();
        if (gm_is_prim_type(typeId) || gm_is_nodeedge_type(typeId))
            continue;
        str = "err = cudaMalloc((void **)&";
        if (onHost)
            str += "h_";
        if (varId->getTypeInfo()->is_graph()) {
            str += std::string(varId->get_orgname());
            str2 = str;
            str += "0";
            str2 += "1";
            str += ", (NumNodes + 2) * sizeof(int));";
            str2 += ", (NumEdges + 1) * sizeof(int));";
            Body.pushln(str.c_str());
            Body.pushln(errCheck.c_str());
            Body.pushln(str2.c_str());
            Body.pushln(errCheck.c_str());
        } else {
            str += std::string(varId->get_orgname()) + ", (" + getSizeOfVariable(varId) + " + 1) * ";
            str += std::string("sizeof(");
            if (varType->is_property()) {
                str += get_type_string(varType->get_target_type());
            }
            else {
                str += get_type_string(varType);
            }
            str += "));";
            Body.pushln(str.c_str());
            Body.pushln(errCheck.c_str());
        }
    }
}

void gm_cuda_gen::CUDAMemcpyToSymbol(ast_node* n) {

    if (n->get_nodetype() == AST_ID) {
        ast_id* id = (ast_id*)n;
        symbol*s = transfer.findSymbol(id->get_orgname());
        std::string str;
        str = "err = cudaMemcpyToSymbol(";
        str += std::string(id->get_orgname()) + ", &";
        if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
            && !printingMacro) {
            str += "h_";
        } else {
            return;
        }
        str += id->get_orgname() + std::string(", ") + std::string("sizeof(");
        str += get_type_string(id->getTypeSummary());
        str += "), 0, cudaMemcpyHostToDevice);";
        Body.pushln(str.c_str());
        str = "CUDA_ERR_CHECK;";
        Body.pushln(str.c_str());
    }/* else if (n->get_nodetype() == AST_FIELD) {
        ast_field* f = (ast_field*)n;
        ast_id* firstField = f->get_first();
        ast_id* secondField = f->get_second();
        CUDAMemcpyToSymbol(firstField);
        //CUDAMemcpyFromSymbol(f->get_first());
        symbol*s = transfer.findSymbol(firstField->get_orgname());
        std::string str;
        str = "err = cudaMemcpy(";
        str += std::string(secondField->get_orgname()) + " + h_" + firstField->get_orgname() + ", ";
        if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
            && !printingMacro) {
            ;//generate;
        } else {
            return;
        }
        str += firstField->get_orgname() + std::string(", ");
        str += getSizeOfVariable(firstField) + " * " + std::string("sizeof(");
        ast_typedecl* varType = firstField->getTypeInfo();
        if (varType->is_property()) {
            str += get_type_string(varType->get_target_type());
        }
        else {
            str += get_type_string(varType);
        }
        str += "), cudaMemcpyHostToDevice);";
        Body.pushln(str.c_str());
        str = "CUDA_ERR_CHECK;";
        Body.pushln(str.c_str());
    }*/
}

void gm_cuda_gen::CUDAMemcpyFromSymbol(ast_node* n) {

    if (n->get_nodetype() == AST_ID) {
        ast_id* id = (ast_id*)n;
        symbol*s = transfer.findSymbol(id->get_orgname());
        std::string str;
        str = "err = cudaMemcpyFromSymbol(&";
        if (!insideCudaKernel && s != NULL && s->getMemLoc() == GPUMemory
            && !printingMacro) {
            str += "h_";
        } else {
            return;
        }
        str += std::string(id->get_orgname()) + ", ";
        str += id->get_orgname() + std::string(", ") + std::string("sizeof(");
        str += get_type_string(id->getTypeSummary());
        str += "), 0, cudaMemcpyDeviceToHost);";
        Body.pushln(str.c_str());
        str = "CUDA_ERR_CHECK;";
        Body.pushln(str.c_str());
    }
}

void gm_cuda_gen::CUDAMemcpy(ast_id* dst, ast_id* dstOffset, std::string src, std::string typeString, bool isHostToDevice) {

    std::string str = "err = cudaMemcpy(";
    str += std::string(dst->get_orgname()) + " + h_" + dstOffset->get_orgname();
    str += ", &" + src + ", " + typeString + ", ";
    if (isHostToDevice)
        str += "cudaMemcpyHostToDevice";
    else
        str += "cudaMemcpyDeviceToHost";
    str += ");";
    Body.pushln(str.c_str());
    Body.pushln("CUDA_ERR_CHECK;");
}

void gm_cuda_gen::CUDAFree(ast_vardecl* varDecl, bool onHost) {
    
    ast_typedecl* varType = varDecl->get_type();
    ast_idlist* idList = varDecl->get_idlist();
    std::string str, str2, errCheck = "CUDA_ERR_CHECK;";
    for (int i = 0; i < idList->get_length(); i++) {
        ast_id* varId = idList->get_item(i);
        if (varId->getSymInfo() == NULL)
            continue;
        int typeId = varId->getTypeSummary();
        if (gm_is_prim_type(typeId) || gm_is_nodeedge_type(typeId))
            continue;
        str = "err = cudaFree(";
        if (onHost)
            str += "h_";
        if (varId->getTypeInfo()->is_graph()) {
            str += std::string(varId->get_orgname());
            str2 = str;
            str += "0);";
            str2 += "1);";
            Body.pushln(str.c_str());
            Body.pushln(errCheck.c_str());
            Body.pushln(str2.c_str());
            Body.pushln(errCheck.c_str());
        } else {
            str += std::string(varId->get_orgname());
            str += ");";
            Body.pushln(str.c_str());
            Body.pushln(errCheck.c_str());
        }
    }
}

void gm_cuda_gen::do_generate_user_main() {
    Body.NL();
    Body.NL();
    Body.pushln("using namespace std;");
    Body.push("// "); Body.push(fname); Body.push(" -? : for how to run generated main program\n");
    Body.pushln("int main(int argc, char* argv[])");
    Body.pushln("{");
    Body.NL();

    Body.pushln("if (argc != 2 || argv[1] == NULL) {");
    Body.pushln("printf(\"Wrong Number of Arguments\");");
    Body.pushln("exit(1);");
    Body.pushln("}");
    Body.pushln("ifstream inputFile;");
    Body.pushln("inputFile.open(argv[1]);");
    Body.pushln("if (!inputFile.is_open()){");
    Body.pushln("printf(\"invalid file\");");
    Body.pushln("exit(1);");
    Body.pushln("}");
    Body.pushln("inputFile >> NumNodes >> NumEdges;");

    Body.pushln("cudaEventCreate(&start);");
    Body.pushln("cudaEventCreate(&stop);");
    Body.pushln("cudaEventRecord(start, 0);");

    assert(FE.get_all_procs().size() == 1);

    std::list<ast_vardecl*>GPUVarList = getGPUScope()->getVarDecls();
    
    for (std::list<ast_vardecl*>::iterator I = GPUVarList.begin();
         I != GPUVarList.end(); I++) {
        CUDAAllocateMemory(*I, true);
    }

    GPUVarList = getGlobalScope()->getVarDecls();
    
    for (std::list<ast_vardecl*>::iterator I = GPUVarList.begin();
         I != GPUVarList.end(); I++) {
        CUDAAllocateMemory(*I, false);
    }
    
    int defaultNumBlocks = 10000;
    std::string str = "bool* host_threadBlockBarrierReached;";
    str = "err = cudaMalloc((void **)&host_threadBlockBarrierReached, ";
    str += "10000 * sizeof(bool));";
    Body.pushln(str.c_str());
    str = "CUDA_ERR_CHECK;";
    Body.pushln(str.c_str());

    str = "err = cudaMemset(host_threadBlockBarrierReached, ";
    str += "0x0, 10000 * sizeof(bool));";
    Body.pushln(str.c_str());
    str = "CUDA_ERR_CHECK;";
    Body.pushln(str.c_str());

    str = "err = cudaMemcpyToSymbol(gm_threadBlockBarrierReached, ";
    str += "&host_threadBlockBarrierReached, sizeof(bool *), 0, ";
    str += "cudaMemcpyHostToDevice);";
    Body.pushln(str.c_str());
    str = "CUDA_ERR_CHECK;";
    Body.pushln(str.c_str());

    Body.pushln("int* h_G[2];");
    Body.pushln("printf(\"Graph Population began\\n\");");
    Body.pushln("populate(argv[1], h_G);");
    Body.pushln("printf(\"Graph Population end\\n\");");

    Body.pushln("cudaEventRecord(stop, 0);");
    Body.pushln("cudaEventSynchronize(stop);");
    Body.pushln("cudaEventElapsedTime(&elapsedTime, start, stop);");
    Body.pushln("printf(\"Loading time(milliseconds)  = %f\\n\", elapsedTime);");


    ast_procdef* proc = FE.get_all_procs()[0];
    ast_typedecl* ret_type = proc->get_return_type();
    std::list<ast_argdecl*>::iterator i;
    std::list<ast_argdecl*>& In = proc->get_in_args();
    std::list<ast_argdecl*>& Out = proc->get_out_args();

    // declare 
    /*for(I=In.begin(); I!= In.end(); I++) {
       print_declare_argument(Body, *I, true, false);
    }
    for(I=Out.begin(); I!= Out.end(); I++) {
       print_declare_argument(Body, *I, false, true);
    }*/
    Body.pushln("cudaEventCreate(&start);");
    Body.pushln("cudaEventCreate(&stop);");
    Body.pushln("cudaEventRecord(start, 0);");
    if (!ret_type->is_void()) {
        assert((ret_type->is_nodeedge()) || (ret_type->is_primitive()));
        str = get_type_string(ret_type);
        str += " MainReturn;";
        Body.pushln(str.c_str());
    }

    char temp[1024];

    if (!ret_type->is_void()) {
        str = "MainReturn";
        Body.push(str.c_str());
        Body.push(" = ");
    }
    str = std::string(proc->get_procname()->get_genname()) + "_CPU";
    Body.push(str.c_str());

    Body.push("(");
    
    bool isFirst = true;
    for (i = In.begin(); i != In.end(); i++) {
        ast_typedecl* type = (*i)->get_type();
        ast_idlist* idlist = (*i)->get_idlist();
        for (int ii = 0; ii < idlist->get_length(); ii++) {
            if (isFirst == false)
                Body.push(", ");
            isFirst = false;

            ast_id* id = idlist->get_item(ii);
            if (type->is_primitive()) {

                sprintf(temp, "%s", id->get_orgname());
            }else if (type->is_graph()) {
                sprintf(temp, "%s0, %s1", id->get_orgname(), id->get_orgname());
            }else if (type->is_property()) {
                sprintf(temp, "%s", id->get_orgname());
            }else if (type->is_nodeedge()) {

                sprintf(temp, "%s", id->get_orgname());
            }else if (type->is_collection()) {

                sprintf(temp, "%s", id->get_orgname());
            }
            Body.push(temp);
        }
    }
    Body.push(");\n");

    Body.NL();
/*    Body.NL();
    Body.pushln("if (!Main.process_arguments(argc, argv)) {");
    Body.pushln("return EXIT_FAILURE;");
    Body.pushln("}");

    Body.NL();
    Body.pushln("if (!Main.do_preprocess()) {");
    Body.pushln("return EXIT_FAILURE;");
    Body.pushln("}");

    Body.pushln("return EXIT_SUCCESS;");*/
    
    Body.pushln("cudaEventRecord(stop, 0);");
    Body.pushln("cudaEventSynchronize(stop);");
    Body.pushln("cudaEventElapsedTime(&elapsedTime, start, stop);");
    Body.pushln("printf(\"Execution time(milliseconds)  = %f\\n\", elapsedTime);");
    
    str = "bool gm_verify = verify" + std::string(proc->get_procname()->get_genname()) + "(h_G);";
    Body.pushln(str.c_str());
    Body.pushln("if (!gm_verify) {");
    Body.pushln("printf(\"Verification Failed\\n\");");
    Body.pushln("return -1;");
    Body.pushln("} else {");
    Body.pushln("printf(\"Verification Success\\n\");");
    Body.pushln("}");
    
    GPUVarList = getGPUScope()->getVarDecls();
    for (std::list<ast_vardecl*>::iterator I = GPUVarList.begin();
         I != GPUVarList.end(); I++) {
        CUDAFree(*I, true);
    }

    GPUVarList = getGlobalScope()->getVarDecls();
    
    for (std::list<ast_vardecl*>::iterator I = GPUVarList.begin();
         I != GPUVarList.end(); I++) {
        CUDAFree(*I, false);
    }
    str = "err = cudaFree(host_threadBlockBarrierReached);";
    Body.pushln(str.c_str());
    str = "CUDA_ERR_CHECK;";
    Body.pushln(str.c_str());
    
    Body.pushln("free(h_G[0]);");
    Body.pushln("free(h_G[1]);");

    if (!ret_type->is_void()) {
        int returnTypeId = ret_type->get_typeid();
        Body.push("printf(\"Return value = ");
        if (gm_is_boolean_type(returnTypeId) || gm_is_integer_type(returnTypeId)) {
            Body.push("%d");
        } else if (gm_is_float_type(returnTypeId)) {
            Body.push("%f");
        }
        Body.push("\\n\", MainReturn);");
        Body.pushln("return MainReturn;");
    }
    Body.pushln("}");
}


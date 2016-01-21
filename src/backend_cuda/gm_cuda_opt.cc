#include <stdio.h>
#include "gm_backend_cuda.h"
#include "gm_error.h"
#include "gm_rw_analysis.h"
#include "gm_typecheck.h"
#include "gm_transform_helper.h"
#include "gm_frontend.h"
#include "gm_ind_opt.h"
#include "gm_argopts.h"
#include "gm_backend_cuda_opt_steps.h"
#include "gm_ind_opt_steps.h"
#include <stack>
#include <fstream>

void gm_cuda_gen::init_opt_steps() {
    std::list<gm_compile_step*>& LIST = this->opt_steps;
    
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_dependencyAnalysis));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_removeAtomicsForBoolean));
/*
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_check_feasible));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_defer));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_common_nbr));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_select_par));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_select_seq_implementation));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_select_map_implementation));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_save_bfs));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_ind_opt_move_propdecl)); // from ind-opt
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_fe_fixup_bound_symbol));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_ind_opt_nonconf_reduce));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_reduce_scalar));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_reduce_field));
    LIST.push_back(GM_COMPILE_STEP_FACTORY(gm_cuda_opt_debug));*/
}

bool gm_cuda_gen::do_local_optimize() {
    // apply all the optimize steps to all procedures
    return gm_apply_compiler_stage(opt_steps);
}

bool gm_cuda_gen::do_local_optimize_lib() {
    //assert(get_lib() != NULL);
    return 1;//get_lib()->do_local_optimize();
}

class BasicBlock {

    std::list<ast_node*> Nodes;
    ast_node* startNode, *endNode;
    std::list<BasicBlock*> succ, pred;
    int id;

public:
    BasicBlock() : startNode(NULL), endNode(NULL) {
        static int basicBlockCount = 0;
        id = basicBlockCount;
        basicBlockCount++;
    }
    int getId() {   return id;  }
    void addSucc(BasicBlock* successor) {
        succ.push_back(successor);
    }
    void addPred(BasicBlock* predecessor) {
        pred.push_back(predecessor);
    }
    std::list<BasicBlock*> getSuccList() {
        return succ;
    }
    std::list<BasicBlock*> getPredList() {
        return pred;
    }
    void addNode(ast_node* n) {
        Nodes.push_back(n);
    }
    std::list<ast_node*> getNodes() {
        return Nodes;
    }

    void dump(std::ofstream &out) {

        out << getId() << " [label = \"" << getId();
        std::list<ast_node*> nodeList = this->getNodes();
        std::list<ast_node*>::iterator nodeListIt = nodeList.begin();
        int instrCount = 0;
        for (; nodeListIt != nodeList.end(); nodeListIt++) {
            ast_node* node = *nodeListIt;
            switch(node->get_nodetype()) {
                case AST_ASSIGN:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    ast_assign* assignNode = (ast_assign*)node;
                    if (assignNode->get_assign_type() == GMASSIGN_NORMAL)
                        out << "<Normal Assign> ";
                    else if (assignNode->get_assign_type() == GMASSIGN_REDUCE) {
                        out << "<Reduce Assign> ";
                    }
                    if (assignNode->get_lhs_type() == GMASSIGN_LHS_SCALA) {
                        out << assignNode->get_lhs_scala()->get_orgname();
                    } else if (assignNode->get_lhs_type() == GMASSIGN_LHS_FIELD) {
                        out << assignNode->get_lhs_field()->get_first()->get_orgname() << ".";
                        out << assignNode->get_lhs_field()->get_second()->get_orgname();
                    }
                    if (assignNode->get_assign_type() == GMASSIGN_REDUCE)
                        out << " " << gm_get_reduce_string(assignNode->get_reduce_type());
                    if (assignNode->is_argminmax_assign())
                        out << " <Multi Assign>";
                    break;
                }
                case AST_VARDECL:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    break;
                }
                case AST_FOREACH:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    ast_foreach* foreachNode = (ast_foreach*)node;
                    if (foreachNode->is_sequential())
                        out << "<For>";
                    else
                        out << "<Foreach>";
                    out << "<It = " << foreachNode->get_iterator()->get_orgname() << ">";
                    out << "<" << foreachNode->get_source()->get_orgname() << ".";
                    out << gm_get_iteration_string(foreachNode->get_iter_type()) << ">";
                    break;
                }
                case AST_IF:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    out << "<IF>";
                    break;
                }
                case AST_WHILE:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    ast_while* whileNode = (ast_while*)node;
                    if (whileNode->is_do_while())
                        out << "<DoWhile>";
                    else
                        out << "<While>";
                    break;
                }
                case AST_RETURN:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    out << "Return";
                    break;
                }
                case AST_CALL:
                {
                    out << "\\n  I" << instrCount++ << " : ";
                    ast_call* callNode = (ast_call*)node;
                    if (callNode->is_builtin_call())
                        out << "<BuiltIn Call>";
                    else
                        out << "<Call>";
                    out << "<" << callNode->getCallName()->get_orgname() << ">";
                    break;
                }
                default:
                    assert(false && "Invalid ast node in the Basic Block");
                    break;
            }
        }
        out << "\"]\n";
    }
};

class ifNodeInfo{
public:
    ifNodeInfo(ast_node* i) : ifNode(i) {
        condBlock = NULL;
        thenBlock = NULL;
        elseBlock = NULL;
    }
    
    ifNodeInfo(ast_node* i, BasicBlock* c) : ifNode(i), condBlock(c) {
        thenBlock = NULL;
        elseBlock = NULL;
    }

    BasicBlock* condBlock, *thenBlock, *elseBlock;
    ast_node* ifNode;

    void setThenBlock(BasicBlock* t) {  thenBlock = t;  }
    void setElseBlock(BasicBlock* e) {  elseBlock = e;  }
};

class CFG : public gm_apply {

    std::list<BasicBlock*> BBList;
    BasicBlock* startBB, *endBB, *currentBB;
    std::stack<BasicBlock*> headerList;
    std::list<ifNodeInfo*> ifNodesStack;
    bool startOfElse;
public:
    CFG() {
        set_for_sent(true);
        set_for_id(false);
        set_separate_post_apply(true);
        endBB = NULL;
        startBB = new BasicBlock();
        currentBB = startBB;
        startOfElse = false;
    }

    void setThenBlock(ast_node* ifNode, BasicBlock* t) {
        
        std::list<ifNodeInfo*>::iterator it = ifNodesStack.begin();
        for (; it != ifNodesStack.end(); it++) {
            ifNodeInfo* currentIfNodeInfo = *it;
            if (currentIfNodeInfo->ifNode == ifNode) {
                currentIfNodeInfo->setThenBlock(t);
                return;
            }
        }
    }
    
    void addSuccToElseBlock(ast_node* ifNode, BasicBlock* e) {

        std::list<ifNodeInfo*>::iterator it = ifNodesStack.begin();
        for (; it != ifNodesStack.end(); it++) {
            ifNodeInfo* currentIfNodeInfo = *it;
            if (currentIfNodeInfo->ifNode == ifNode) {
                currentIfNodeInfo->condBlock->addSucc(e);
                return;
            }
        }
    }

    bool apply(ast_sent* s) {
        if (s->get_parent() != NULL && s->get_nodetype() != AST_SENTBLOCK) {
            bool insideSentBlock = false;
            ast_node* parent = s->get_parent();
            if (parent->get_nodetype() == AST_SENTBLOCK) {
                parent = parent->get_parent();
                insideSentBlock = true;
            }
            if (parent->get_nodetype() == AST_IF) {
                ast_if* parentIfNode = (ast_if*)parent;
                if (insideSentBlock) {
                    ast_sentblock* elseSentBlock = (ast_sentblock*)(parentIfNode->get_else());
                    if (elseSentBlock != NULL) {
                        std::list<ast_sent*> elseSentBlockStmtList = elseSentBlock->get_sents();
                        ast_node* firstNodeInElseBlock = (ast_node*) (elseSentBlockStmtList.front());
                        if (s == firstNodeInElseBlock) {
                            startOfElse = true;
                            setThenBlock(parent, currentBB);
                            BasicBlock* newBB = new BasicBlock();
                            addSuccToElseBlock(parent, newBB);
                            currentBB = newBB;
                        }
                    }
                } else {
                    if (parentIfNode->get_else() == s) {
                        startOfElse = true;
                        setThenBlock(parent, currentBB);
                        BasicBlock* newBB = new BasicBlock();
                        addSuccToElseBlock(parent, newBB);
                        currentBB = newBB;
                    }
                }
            }
        }

        if (s->get_nodetype() == AST_FOREACH || 
            s->get_nodetype() == AST_WHILE) {
            BasicBlock* newBB;
            // Basic Block for Foreach Entry(Header)
            if (!startOfElse) {
                newBB = new BasicBlock();
                currentBB->addSucc(newBB);
            } else {
                newBB = currentBB;
                startOfElse = false;
            }
            headerList.push(newBB);
            currentBB = newBB;
            currentBB->addNode((ast_node*)s);
            // Basic Block for Foreach Body
            newBB = new BasicBlock();
            currentBB->addSucc(newBB);
            currentBB = newBB;
        } else if (s->get_nodetype() == AST_IF) {
            currentBB->addNode((ast_node*)s);
            BasicBlock* thenBlock = new BasicBlock();
            currentBB->addSucc(thenBlock);
            ifNodeInfo* newIfNode = new ifNodeInfo(s, currentBB);
            currentBB = thenBlock;
            ifNodesStack.push_back(newIfNode);
        } else if (s->get_nodetype() == AST_CALL) {
            currentBB->addNode((ast_node*)s);
        } else if (s->get_nodetype() == AST_ASSIGN ||
                   s->get_nodetype() == AST_VARDECL ||
                   s->get_nodetype() == AST_RETURN){
            currentBB->addNode((ast_node*)s);
        }
    }

    bool apply2(ast_sent* s) {
        if (s->get_nodetype() == AST_FOREACH || 
            s->get_nodetype() == AST_WHILE) {
            assert(headerList.size() > 0);
            BasicBlock* headerBB = headerList.top();
            headerList.pop();
            currentBB->addSucc(headerBB);
            BasicBlock* newBB = new BasicBlock();
            headerBB->addSucc(newBB);
            currentBB = newBB;
        } else if (s->get_nodetype() == AST_IF) {
            ast_if* currentIfNode = (ast_if*) s;
            ifNodeInfo* nodeInfo = ifNodesStack.back();
            ifNodesStack.pop_back();
            if (nodeInfo->ifNode != s) {
                assert(false && "Current IfNode is not in the stack");
            }
            if (nodeInfo->thenBlock == NULL)
                nodeInfo->thenBlock = currentBB;
            BasicBlock* thenBlock = nodeInfo->thenBlock;
            BasicBlock* newBB = new BasicBlock();
            thenBlock->addSucc(newBB);
            if (currentIfNode->get_else() == NULL) {
                BasicBlock* condBlock = nodeInfo->condBlock;
                condBlock->addSucc(newBB);
            } else {
                BasicBlock* elseBlock = currentBB;
                elseBlock->addSucc(newBB);
            }
            currentBB = newBB;
        }
    }

    void printCFG() {
        std::ofstream cfgFile;
        cfgFile.open("cfg.dot", std::ofstream::out);
        cfgFile << "strict digraph cfg {\n";
        std::map<BasicBlock*, bool> printedBlocks;
        std::list<BasicBlock*> bbList;
        bbList.push_back(startBB);
        while(!bbList.empty()) {
            BasicBlock* bb = bbList.back();
            bbList.pop_back();
            if (!printedBlocks[bb]) {
                bb->dump(cfgFile);
                printedBlocks[bb] = true;
            }
            std::list<BasicBlock*> succList = bb->getSuccList();
            std::list<BasicBlock*>::iterator succIt = succList.begin();
            for (; succIt != succList.end(); succIt++) {
                BasicBlock* succ = *succIt;
                printf("%d -> %d\n", bb->getId(), succ->getId());
                cfgFile << bb->getId() << " -> " << succ->getId() << "\n";
                if (succ->getId() > bb->getId())
                    bbList.push_back(succ);
            }
        }
        cfgFile << "}\n";
        cfgFile.close();
    }
};

void gm_cuda_opt_dependencyAnalysis::process(ast_procdef* proc) {

    CFG currentCFG;
    proc->traverse_both(&currentCFG);
    currentCFG.printCFG();
}

class gm_identifyBooleanReduction : public gm_apply {

public:
    gm_identifyBooleanReduction() {
        set_for_sent(true);
    }

    bool apply(ast_sent* s) {

    }
};

// For reduction statement of Boolean type, we eliminate atomics and replace with
// a normal assignment statement.
void gm_cuda_opt_removeAtomicsForBoolean::process(ast_procdef* proc) {

    //gm_identifyBooleanReduction(proc->get_body());
}


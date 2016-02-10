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
#include <string>
#include <stack>
#include <fstream>

using namespace std;
void gm_cuda_gen::init_opt_steps() {
    list<gm_compile_step*>& LIST = this->opt_steps;
    
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

struct compareNodes {
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
    
    bool operator() (ast_node*& n1, ast_node*& n2) {
//    bool sameVariableName(ast_node* n1, ast_node* n2) {
    
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
};

class Def;
//typedef map<ast_node*, list<Def *>, compareNodes > mapASTNodeToDefs;
typedef map<string , list<Def *> > mapVarToDefs;
class BasicBlock {

    list<ast_node*> Nodes;
    ast_node* startNode, *endNode;
    list<BasicBlock*> succ, pred;
    int id;

    list<Def*> IN, OUT;
    mapVarToDefs defsList;

public:
    BasicBlock() : startNode(NULL), endNode(NULL) {
        static int basicBlockCount = 0;
        id = basicBlockCount;
        basicBlockCount++;
    }
    int getId() {   return id;  }
    void addSucc(BasicBlock* successor) {
        succ.push_back(successor);
       successor-> addPred(this);
    }
    void addPred(BasicBlock* predecessor) {
        pred.push_back(predecessor);
    }
    list<BasicBlock*> getSuccList() {
        return succ;
    }
    list<BasicBlock*> getPredList() {
        return pred;
    }
    void addNode(ast_node* n) {
        Nodes.push_back(n);
    }
    list<ast_node*> getNodes() {
        return Nodes;
    }

    mapVarToDefs getVarDefsList() { return defsList;    }
    void addToDefsList(ast_node* n);
    void addToDefsList(Def* d);
    void propogateDefs(BasicBlock* succ);

    void dump(ofstream &out) {

        out << getId() << " [label = \"" << getId();
        list<ast_node*> nodeList = this->getNodes();
        list<ast_node*>::iterator nodeListIt = nodeList.begin();
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

    list<BasicBlock*> BBList;
    BasicBlock* startBB, *endBB, *currentBB;
    stack<BasicBlock*> headerList;
    list<ifNodeInfo*> ifNodesStack;
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

    BasicBlock* getStartBB() {  return startBB; }
    void setThenBlock(ast_node* ifNode, BasicBlock* t) {
        
        list<ifNodeInfo*>::iterator it = ifNodesStack.begin();
        for (; it != ifNodesStack.end(); it++) {
            ifNodeInfo* currentIfNodeInfo = *it;
            if (currentIfNodeInfo->ifNode == ifNode) {
                currentIfNodeInfo->setThenBlock(t);
                return;
            }
        }
    }
    
    void addSuccToElseBlock(ast_node* ifNode, BasicBlock* e) {

        list<ifNodeInfo*>::iterator it = ifNodesStack.begin();
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
                        list<ast_sent*> elseSentBlockStmtList = elseSentBlock->get_sents();
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
        ofstream cfgFile;
        cfgFile.open("cfg.dot", ofstream::out);
        cfgFile << "strict digraph cfg {\n";
        map<BasicBlock*, bool> printedBlocks;
        list<BasicBlock*> bbList;
        bbList.push_back(startBB);
        while(!bbList.empty()) {
            BasicBlock* bb = bbList.back();
            bbList.pop_back();
            if (!printedBlocks[bb]) {
                bb->dump(cfgFile);
                printedBlocks[bb] = true;
            }
            list<BasicBlock*> succList = bb->getSuccList();
            list<BasicBlock*>::iterator succIt = succList.begin();
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

class Def {
    ast_node* node;
    BasicBlock* enclosingBlock;

public:
    Def(ast_node* n) : node(n) { 
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
        enclosingBlock = NULL;
    }
    Def(ast_node* n, BasicBlock* b) : node(n), enclosingBlock(b) {
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
    }

    ast_node* getNode() {   return node;    }
    BasicBlock* getBB() {   return enclosingBlock;  }
};

void BasicBlock::addToDefsList(ast_node* n) {
    Def* newDef = new Def(n, this);
    string nodeName;
    if (n->get_nodetype() == AST_ID)
        nodeName = ((ast_id*)n)->get_orgname();
    else if (n->get_nodetype() == AST_FIELD) {
        ast_field* fieldNode = (ast_field*)n;
        nodeName = /*fieldNode->get_first()->get_orgname() + string(".") +*/ fieldNode->get_second()->get_orgname();
    }
    mapVarToDefs::iterator nodeDefList = defsList.find(nodeName);
    if (nodeDefList == defsList.end()) {
        defsList[nodeName].push_back(newDef);
    } else {
        (*nodeDefList).second.push_back(newDef);
    }
}

void BasicBlock::addToDefsList(Def* newDef) {
    ast_node* n = newDef->getNode();
    string nodeName;
    if (n->get_nodetype() == AST_ID)
        nodeName = ((ast_id*)n)->get_orgname();
    else if (n->get_nodetype() == AST_FIELD) {
        ast_field* fieldNode = (ast_field*)n;
        nodeName = /*fieldNode->get_first()->get_orgname() + string(".") + */fieldNode->get_second()->get_orgname();
    }
    mapVarToDefs::iterator nodeDefList = defsList.find(nodeName);
    if (nodeDefList == defsList.end()) {
        defsList[nodeName].push_back(newDef);
    } else {
        (*nodeDefList).second.push_back(newDef);
    }
}

void BasicBlock::propogateDefs(BasicBlock* succBB) {
    mapVarToDefs varDefs = getVarDefsList();
    mapVarToDefs::iterator varDefIt = varDefs.begin();
    for (; varDefIt != varDefs.end(); varDefIt++) {
        //printf("\tVar: %s. ", (*varDefIt).first.c_str());
        string nodeName = (*varDefIt).first;
        list<Def*>::iterator defIt = (*varDefIt).second.begin();
        for (; defIt != (*varDefIt).second.end(); defIt++) {
            Def* oldDef = *defIt;
            succBB->addToDefsList(oldDef);
        }
    }
}

class Use {
    ast_node* node;
    BasicBlock* enclosingBlock;
    list<Def*> reachingDef;

public:
    Use(ast_node* n) : node(n) { 
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
        enclosingBlock = NULL;
    }
    Use(ast_node* n, BasicBlock* b) : node(n), enclosingBlock(b) {
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
    }

    ast_node* getNode() {   return node;    }
    BasicBlock* getBB() {   return enclosingBlock;  }
    list<Def*> getReachingDefs() {  return reachingDef; }

    void addDefList(list<Def*> newDefList) {
        for (list<Def*>::iterator defIt = newDefList.begin(); defIt != newDefList.end(); defIt++) {
            Def* newDef = *defIt;
            bool alreadyPresent = false;
            for (list<Def*>::iterator listIt = reachingDef.begin(); listIt != reachingDef.end(); listIt++) {
                Def* listDef = *listIt;
                if (newDef == listDef) {
                    alreadyPresent = true;
                    break;
                }
            }
            if (!alreadyPresent)
                reachingDef.push_back(newDef);
        }
    }
    void addDefToReachingDef(Def* newDef) {
        reachingDef.push_back(newDef);
    }
};

typedef pair<ast_foreach*, BasicBlock*> foreachBBPair;
class DefUseAnalysis : gm_apply {

    CFG* currentCFG;

    BasicBlock* currentBB;
    list<BasicBlock*> worklist;
    list<foreachBBPair> foreachToBBMap;
//    map<string, Def*> mapNodeToDef;
    map<ast_node*, Use*> mapNodeToUse;

    bool propogatePhase;
public:
    DefUseAnalysis(CFG* c) {
        set_for_sent(true);
        set_for_id(false);
        set_separate_post_apply(true);
        currentCFG = c;
        propogatePhase = false;
    }

    void addArgDefs(ast_procdef* proc) {
        BasicBlock* startBB = currentCFG->getStartBB();
        std::list<ast_argdecl*>& inArgs = proc->get_in_args();
        std::list<ast_argdecl*>::iterator i;
        for (i = inArgs.begin(); i != inArgs.end(); i++) {
            ast_typedecl* type = (*i)->get_type();
            ast_idlist* idlist = (*i)->get_idlist();
            for (int ii = 0; ii < idlist->get_length(); ii++) {
                ast_id* id = idlist->get_item(ii);
                Def* newDef = new Def(id, currentBB);
                startBB->addToDefsList(newDef);
                //string nodeName = id->get_orgname();
                //mapNodeToDef[nodeName] = newDef;
            }
        }
    }
    void setPropogatePhase(bool b) {
        propogatePhase = b;
    }

    bool apply(ast_sent* s) {
        if (s->get_nodetype() == AST_ASSIGN) {
            ast_assign* assignSent = (ast_assign*)s;

            if (assignSent->is_reduce_assign() && !propogatePhase) {
                ast_id* boundIt = assignSent->get_bound();
                foreachBBPair pair;
                bool found = false;
                for (list<foreachBBPair>::iterator It = foreachToBBMap.begin(); It != foreachToBBMap.end(); It++) {
                    pair = *It;
                    if (!strcmp(pair.first->get_iterator()->get_orgname(), boundIt->get_orgname())) {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    assert(false && "Bounding iterator for the reduction assignment not found");

                BasicBlock* foreachBB = pair.second, *BBAfterForeach = NULL;
                list<BasicBlock*> succList = foreachBB->getSuccList();
                /*for (list<BasicBlock*>::iterator It = succList.begin(); It != succList.end(); It++) {
                    if (foreachBB->getId() < (*It)->getId()) {
                        if (BBAfterForeach != NULL)
                            assert(false && "More than one successor for foreach loop");
                        BBAfterForeach = *It;
                    }
                }*/
                BBAfterForeach = succList.back();

                if (BBAfterForeach == NULL)
                    assert(false && "Could not find a successor of foreach loop");

                if (assignSent->is_target_scalar()) {
                    ast_id* target = assignSent->get_lhs_scala();
                    BBAfterForeach->addToDefsList(target);
                } else {
                    ast_field* targetField = assignSent->get_lhs_field();
                    BBAfterForeach->addToDefsList(targetField);
                }
                if (assignSent->has_lhs_list() && assignSent->is_argminmax_assign()) {
                    list<ast_node*> lhsList = assignSent->get_lhs_list();
                    list<ast_node*>::iterator lhsListIt = lhsList.begin();
                    for (; lhsListIt != lhsList.end(); lhsListIt++) {
                        ast_node* targetNode = *lhsListIt;
                        BBAfterForeach->addToDefsList(targetNode);
                    }
                }
            } else if (!propogatePhase){
                Def* newDef;
                if (assignSent->is_target_scalar()) {
                    ast_node* targetNode = assignSent->get_lhs_scala();
                    newDef = new Def(targetNode, currentBB);
                    currentBB->addToDefsList(newDef);
                    //string nodeName = ((ast_id*)targetNode)->get_orgname();
                    //mapNodeToDef[nodeName] = newDef;
                } else {
                    ast_node* targetNode = assignSent->get_lhs_field();
                    newDef = new Def(targetNode, currentBB);
                    currentBB->addToDefsList(newDef);
                    ast_field* fieldNode = (ast_field*)targetNode;
                    //string nodeName = fieldNode->get_first()->get_orgname() + string(".") + fieldNode->get_second()->get_orgname();
                    //mapNodeToDef[nodeName] = newDef;
                }
            }
            if (propogatePhase) {
                set_for_id(true);
                assignSent->get_rhs()->traverse_pre(this);
                if (assignSent->is_reduce_assign() && assignSent->has_lhs_list() && assignSent->is_argminmax_assign()) {
                    list<ast_expr*> rhsList = assignSent->get_rhs_list();
                    list<ast_expr*>::iterator rhsListIt = rhsList.begin();
                    for (; rhsListIt != rhsList.end(); rhsListIt++) {
                        ast_expr* rhsExpr = *rhsListIt;
                        rhsExpr->traverse_pre(this);
                    }
                }
                set_for_id(false);
            }
        } else if (s->get_nodetype() == AST_FOREACH) {
            ast_foreach* foreachSent = (ast_foreach*)s;
            if (!propogatePhase) {
                Def* newDef;
                ast_node* iterNode = foreachSent->get_iterator();
                newDef = new Def(iterNode, currentBB);
                currentBB->addToDefsList(newDef);
                //string nodeName = ((ast_id*)iterNode)->get_orgname();
                //mapNodeToDef[nodeName] = newDef;
            } else if (propogatePhase) {
                set_for_id(true);
                foreachSent->get_source()->traverse_pre(this);
                set_for_id(false);
            }
        }
    }

    bool apply(ast_id* id) {
        ast_node* parent = id->get_parent();
        string nodeName = id->get_orgname();
        mapVarToDefs BBDefsList = currentBB->getVarDefsList();
        list<Def*> varDefList;
        mapVarToDefs::iterator nodeDefList = BBDefsList.find(nodeName);
        if (nodeDefList != BBDefsList.end()) {
            varDefList = BBDefsList[nodeName];
        } else {
            //(*nodeDefList).second.push_back(newDef);
            assert(false && "Variable is not defined yet.");
        }
        if (parent == NULL) {
            map<ast_node*, Use*>::iterator it = mapNodeToUse.find(id);
            if (it == mapNodeToUse.end()) {
                Use* newUse = new Use(id, currentBB);
                newUse->addDefList(varDefList);
                mapNodeToUse[id] = newUse;
            } else {
                (*it).second->addDefList(varDefList);
            }
        }
        if (parent->get_nodetype() == AST_FIELD) {
            ast_field* fieldNode = (ast_field*)parent;
            if (fieldNode->get_first() == id) {
                ; // do Nothing
            } else if (fieldNode->get_second() == id) {
                ;
            }
        } else if (parent->get_nodetype() == AST_MAPACCESS) {
            ;
        } else {
            ;
        }
    }

    void processBlock(BasicBlock* bb) {
        list<ast_node*> nodeList = bb->getNodes();

        for (list<ast_node*>::iterator nodeListIt = nodeList.begin();
                nodeListIt != nodeList.end(); nodeListIt++) {
            ast_node* node = *nodeListIt;
            currentBB = bb;
            if (node->get_nodetype() == AST_FOREACH ||
                node->get_nodetype() == AST_WHILE) {
                ast_foreach* foreachSent = (ast_foreach*)node;
                foreachBBPair newForeach(foreachSent, currentBB);
                foreachToBBMap.push_front(newForeach);
            } else if (node->get_nodetype() != AST_IF) {
                node->traverse_pre(this);
            }

            if (node->get_nodetype() == AST_FOREACH) {
                ast_foreach* foreachSent = (ast_foreach*)node;
                if (!propogatePhase) {
                    Def* newDef;
                    ast_node* iterNode = foreachSent->get_iterator();
                    newDef = new Def(iterNode, currentBB);
                    currentBB->addToDefsList(newDef);
                    //string nodeName = ((ast_id*)iterNode)->get_orgname();
                    //mapNodeToDef[nodeName] = newDef;
                } else if (propogatePhase) {
                    set_for_id(true);
                    ast_id* sourceNode = foreachSent->get_source();
                    string nodeName = sourceNode->get_orgname();
                    mapVarToDefs BBDefsList = currentBB->getVarDefsList();
                    list<Def*> varDefList;
                    mapVarToDefs::iterator nodeDefList = BBDefsList.find(nodeName);
                    if (nodeDefList != BBDefsList.end()) {
                        varDefList = BBDefsList[nodeName];
                    } else {
                        //(*nodeDefList).second.push_back(newDef);
                        assert(false && "Variable is not defined yet.");
                    }
                    map<ast_node*, Use*>::iterator it = mapNodeToUse.find(sourceNode);
                    if (it == mapNodeToUse.end()) {
                        Use* newUse = new Use(sourceNode, currentBB);
                        newUse->addDefList(varDefList);
                        mapNodeToUse[sourceNode] = newUse;
                    } else {
                        (*it).second->addDefList(varDefList);
                    }
                    set_for_id(false);
                }
            }
        }
    }

    void analyse() {

        map<int, bool> visited;
        worklist.push_back(currentCFG->getStartBB());
        visited[worklist.front()->getId()] = true;
        while(!worklist.empty()) {
            BasicBlock* bb = worklist.front();
            worklist.pop_front();
            printf("BBNumber : %d\n", bb->getId());
            processBlock(bb);
            list<BasicBlock*>succList = bb->getSuccList();
            for (list<BasicBlock*>::iterator succListIt = succList.begin(); 
                    succListIt != succList.end(); succListIt++) {
                BasicBlock* succBB = *succListIt;
                {
                    bb->propogateDefs(succBB);
                    // Insert the New Basic Block into the sorted Linked list.
                    bool isFound = false;
                    list<BasicBlock*>::iterator worklistIt = worklist.begin();
                    for (; worklistIt != worklist.end(); worklistIt++) {
                        BasicBlock* itBB = *worklistIt;
                        if (itBB->getId() > succBB->getId())
                            break;
                        if (itBB->getId() == succBB->getId()) {
                            isFound = true;
                            break;
                        }
                    }
                    if (!isFound && (visited.find(succBB->getId()) == visited.end())) {
                        worklist.insert(worklistIt, succBB);
                        visited[succBB->getId()] = true;
                    }
                }
            }
        }
    }
    void print() {
        
        map<int, bool> visited;
        worklist.push_back(currentCFG->getStartBB());
        visited[worklist.front()->getId()] = true;
        while(!worklist.empty()) {
            BasicBlock* bb = worklist.front();
            worklist.pop_front();
            printf("\nBBNumber : %d\n", bb->getId());

            mapVarToDefs varDefs = bb->getVarDefsList();
            mapVarToDefs::iterator varDefIt = varDefs.begin();
            for (; varDefIt != varDefs.end(); varDefIt++) {
                printf("\tVar: %s. ", (*varDefIt).first.c_str());
            }

            list<BasicBlock*>succList = bb->getSuccList();
            for (list<BasicBlock*>::iterator succListIt = succList.begin(); 
                    succListIt != succList.end(); succListIt++) {
                BasicBlock* succBB = *succListIt;
                {
                    // Insert the New Basic Block into the sorted Linked list.
                    bool isFound = false;
                    list<BasicBlock*>::iterator worklistIt = worklist.begin();
                    for (; worklistIt != worklist.end(); worklistIt++) {
                        BasicBlock* itBB = *worklistIt;
                        if (itBB->getId() > succBB->getId())
                            break;
                        if (itBB->getId() == succBB->getId()) {
                            isFound = true;
                            break;
                        }
                    }
                    if (!isFound && (visited.find(succBB->getId()) == visited.end())) {
                        worklist.insert(worklistIt, succBB);
                        visited[succBB->getId()] = true;
                    }
                }
            }
        }
    }
};

class printDefUseChain : public gm_apply {

public:
    printDefUseChain() {

    }
    bool apply(ast_sent* s) {

    }

    bool apply(ast_id* id) {

    }
}
void gm_cuda_opt_dependencyAnalysis::process(ast_procdef* proc) {

    CFG currentCFG;
    proc->traverse_both(&currentCFG);
    currentCFG.printCFG();
    DefUseAnalysis analysis(&currentCFG);
    analysis.addArgDefs(proc);
    analysis.analyse();
    analysis.setPropogatePhase(true);
    analysis.analyse();
    analysis.print();
    //proc->traverse_pre(&analysis);
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


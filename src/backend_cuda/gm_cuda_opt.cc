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

class Def;
typedef map<string , list<Def *> > mapVarToDefs;
class BasicBlock {

    list<ast_node*> Nodes;
    ast_node* startNode, *endNode;
    list<BasicBlock*> succ, pred;
    int id;

    mapVarToDefs IN, OUT, genSet, killSet;

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

    mapVarToDefs getGenSet() {  return genSet;  }
    bool addToGenSet(ast_sent* s, ast_node* n);
    bool addToGenSet(Def* d);

    mapVarToDefs getKillSet() { return killSet; }
    bool addToKillSet(Def* d);

    void printGenKillSet();
    void printINAndOUTSets();

    mapVarToDefs getINSet() { return IN; }
    bool addToINSet(Def* inDef);
    bool propogateOutToSucc(BasicBlock* succBB);
    bool generateOut();

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
    map<ast_sent*, BasicBlock*> mapSentToBB;
    bool startOfElse;
public:
    CFG() {
        set_for_sent(true);
        set_for_id(false);
        set_separate_post_apply(true);
        endBB = NULL;
        startBB = new BasicBlock();
        addToBBList(startBB);
        currentBB = startBB;
        startOfElse = false;
    }

    void addToBBList(BasicBlock* newBB) {
        BBList.push_back(newBB);
    }

    BasicBlock* getBBForSent(ast_sent* s) {
        map<ast_sent*, BasicBlock*>::iterator sentToBBIt = mapSentToBB.find(s);
        if (sentToBBIt == mapSentToBB.end())
            return NULL;
        return (*sentToBBIt).second;
    }
    
    list<BasicBlock*> getBBList() { return BBList;  }

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
                            addToBBList(newBB);
                            addSuccToElseBlock(parent, newBB);
                            currentBB = newBB;
                        }
                    }
                } else {
                    if (parentIfNode->get_else() == s) {
                        startOfElse = true;
                        setThenBlock(parent, currentBB);
                        BasicBlock* newBB = new BasicBlock();
                        addToBBList(newBB);
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
                addToBBList(newBB);
                currentBB->addSucc(newBB);
            } else {
                newBB = currentBB;
                startOfElse = false;
            }
            headerList.push(newBB);
            currentBB = newBB;
            currentBB->addNode((ast_node*)s);
            mapSentToBB[s] = currentBB;
            // Basic Block for Foreach Body
            newBB = new BasicBlock();
            addToBBList(newBB);
            currentBB->addSucc(newBB);
            currentBB = newBB;
        } else if (s->get_nodetype() == AST_IF) {
            currentBB->addNode((ast_node*)s);
            mapSentToBB[s] = currentBB;
            BasicBlock* thenBlock = new BasicBlock();
            addToBBList(thenBlock);
            currentBB->addSucc(thenBlock);
            ifNodeInfo* newIfNode = new ifNodeInfo(s, currentBB);
            currentBB = thenBlock;
            ifNodesStack.push_back(newIfNode);
        } else if (s->get_nodetype() == AST_CALL) {
            currentBB->addNode((ast_node*)s);
            mapSentToBB[s] = currentBB;
        } else if (s->get_nodetype() == AST_ASSIGN ||
                   s->get_nodetype() == AST_VARDECL ||
                   s->get_nodetype() == AST_RETURN){
            currentBB->addNode((ast_node*)s);
            mapSentToBB[s] = currentBB;
        }
        mapSentToBB[s] = currentBB;
    }

    bool apply2(ast_sent* s) {
        if (s->get_nodetype() == AST_FOREACH || 
            s->get_nodetype() == AST_WHILE) {
            assert(headerList.size() > 0);
            BasicBlock* headerBB = headerList.top();
            headerList.pop();
            currentBB->addSucc(headerBB);
            BasicBlock* newBB = new BasicBlock();
            addToBBList(newBB);
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
            addToBBList(newBB);
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
                //printf("%d -> %d\n", bb->getId(), succ->getId());
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
    ast_sent* defStmt;
    ast_node* node;
    BasicBlock* enclosingBlock;

public:
    Def(ast_sent* s, ast_node* n) : defStmt(s), node(n) { 
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
        enclosingBlock = NULL;
    }
    Def(ast_sent* s, ast_node* n, BasicBlock* b) : defStmt(s), node(n), enclosingBlock(b) {
        if (!(n->get_nodetype() == AST_ID || n->get_nodetype() == AST_FIELD))
            assert(false && "Adding variable of different type for the definition");
    }

    ast_sent* getStmt() {   return defStmt; }
    ast_node* getNode() {   return node;    }
    BasicBlock* getBB() {   return enclosingBlock;  }

    char* getName() {
        if (node->get_nodetype() == AST_ID) {
            return ((ast_id*)node)->get_orgname();
        } else if (node->get_nodetype() == AST_FIELD) {
            ast_field* fieldNode = (ast_field*)node;
            ast_id* property = fieldNode->get_second();
            return property->get_orgname();
        }
    }
};

bool addToMapDefList(mapVarToDefs* insertInto, Def* newDef, bool eraseDefs = false) {

    string nodeName = newDef->getName();
    mapVarToDefs::iterator mapVarToDefsIt = insertInto->find(nodeName);
    if (mapVarToDefsIt == insertInto->end()) {
        (*insertInto)[nodeName].push_back(newDef);
        return true;
    } else {
        for (list<Def*>::iterator defListIt = (*mapVarToDefsIt).second.begin();
            defListIt != (*mapVarToDefsIt).second.end(); defListIt++) {
            if (newDef == *defListIt)
                return false;
        }
        if (eraseDefs)
            (*mapVarToDefsIt).second.clear();
        (*mapVarToDefsIt).second.push_back(newDef);
        return true;
    }
}

bool BasicBlock::addToGenSet(ast_sent* s, ast_node* n) {
    Def* newDef = new Def(s, n, this);
    return addToGenSet(newDef);
}

bool BasicBlock::addToGenSet(Def* newDef) {
    return addToMapDefList(&genSet, newDef, true);
}

bool BasicBlock::addToKillSet(Def* killDef) {
    return addToMapDefList(&killSet, killDef);
}

void BasicBlock::printGenKillSet() {
    int genCount = 0;
    printf("Generated definition for variable \n");
    mapVarToDefs::iterator defListIt = genSet.begin();
    for (; defListIt != genSet.end(); defListIt++) {
        string nodeName = (*defListIt).first;
        list<Def *>defs = (*defListIt).second;
        for (list<Def *>::iterator defIt = defs.begin(); defIt != defs.end(); defIt++) {
            Def* nodeDef = *defIt;
            printf("\t%d : %s from stmt\n", genCount, nodeDef->getName());
            if (nodeDef->getStmt() != NULL) {
                nodeDef->getStmt()->dump_tree(2);
                printf("\t\tFrom Basic Block %d\n", nodeDef->getBB()->getId());
            } else {
                printf("\t\tAs Argument\n");
            }
            genCount++;
        }
        printf("\n\n");
    }

    int killCount = 0;
    printf("Killed definition for variable \n");
    defListIt = killSet.begin();
    for (; defListIt != killSet.end(); defListIt++) {
        string nodeName = (*defListIt).first;
        list<Def *>defs = (*defListIt).second;
        for (list<Def *>::iterator defIt = defs.begin(); defIt != defs.end(); defIt++) {
            Def* killDef = *defIt;
            printf("\t%d : %s from stmt\n", killCount, killDef->getName());
            if (killDef->getStmt() == NULL)
                printf("\t\tAs Argument\n");
            else {
                killDef->getStmt()->dump_tree(2);
                printf("\t\tFrom Basic Block %d\n", killDef->getBB()->getId());
            }
            killCount++;
        }
        printf("\n\n");
    }
}

void BasicBlock::printINAndOUTSets() {
    int count = 0;
    printf("IN Set::\n");
    mapVarToDefs::iterator defListIt = IN.begin();
    for (; defListIt != IN.end(); defListIt++) {
        string nodeName = (*defListIt).first;
        list<Def *>defs = (*defListIt).second;
        for (list<Def *>::iterator defIt = defs.begin(); defIt != defs.end(); defIt++) {
            Def* nodeDef = *defIt;
            printf("\t%d : %s from stmt\n", count, nodeDef->getName());
            if (nodeDef->getStmt() != NULL) {
                nodeDef->getStmt()->dump_tree(2);
                printf("\t\tFrom Basic Block %d\n", nodeDef->getBB()->getId());
            } else {
                printf("\t\tAs Argument\n");
            }
            count++;
        }
        printf("\n\n");
    }

    count = 0;
    printf("OUT Set::\n");
    defListIt = OUT.begin();
    for (; defListIt != OUT.end(); defListIt++) {
        string nodeName = (*defListIt).first;
        list<Def *>defs = (*defListIt).second;
        for (list<Def *>::iterator defIt = defs.begin(); defIt != defs.end(); defIt++) {
            Def* nodeDef = *defIt;
            printf("\t%d : %s from stmt\n", count, nodeDef->getName());
            if (nodeDef->getStmt() != NULL) {
                nodeDef->getStmt()->dump_tree(2);
                printf("\t\tFrom Basic Block %d\n", nodeDef->getBB()->getId());
            } else {
                printf("\t\tAs Argument\n");
            }
            count++;
        }
        printf("\n\n");
    }
}

bool BasicBlock::addToINSet(Def* inDef) {

    return addToMapDefList(&IN, inDef);
}

bool BasicBlock::propogateOutToSucc(BasicBlock* succBB) {

    bool hasChanged = false;
    mapVarToDefs::iterator outIt = OUT.begin();
    for (; outIt != OUT.end(); outIt++) {
        string varName = (*outIt).first;
        list<Def*> outDefList = (*outIt).second;
        for (list<Def*>::iterator outDefListIt = outDefList.begin();
            outDefListIt != outDefList.end(); outDefListIt++) {
        
            hasChanged |= succBB->addToINSet(*outDefListIt);
        }
    }
    return hasChanged;
}

bool BasicBlock::generateOut() {

    bool hasChanged = false;
    mapVarToDefs BBGenSet = getGenSet();
    for (mapVarToDefs::iterator genSetIt = BBGenSet.begin(); genSetIt != BBGenSet.end(); genSetIt++) {
        list<Def*> varDefList = (*genSetIt).second;

        for (list<Def*>::iterator genSetIt = varDefList.begin();
            genSetIt != varDefList.end(); genSetIt++) {
            
            Def* genDef = (*genSetIt);
            hasChanged |= addToMapDefList(&OUT, genDef);
        }
    }
    
    for (mapVarToDefs::iterator INSetIt = IN.begin(); INSetIt != IN.end(); INSetIt++) {
        string varName = (*INSetIt).first;
        list<Def*> INVarDefList = (*INSetIt).second;

        mapVarToDefs::iterator killSetIt = killSet.find(varName);
        for (list<Def*>::iterator INVarDefListIt = INVarDefList.begin();
            INVarDefListIt != INVarDefList.end(); INVarDefListIt++) {
            
            Def* INDef = (*INVarDefListIt);
            bool killTheINDef= false;
            if (killSetIt == killSet.end()) {
                ;
            } else {
                list<Def*> killDefList = (*killSetIt).second;
                for (list<Def*>::iterator killDefListIt = killDefList.begin();
                    killDefListIt != killDefList.end(); killDefListIt++) {
                
                    if (INDef == *killDefListIt) {
                        killTheINDef = true;
                        break;
                    }
                }
            }
            if (killTheINDef == false) {
                hasChanged |= addToMapDefList(&OUT, INDef);
            }
        }
    }
    return hasChanged;
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

    void printReachingDefs() {
        printf("Defined at :");
        int defCount = 0;
        for (list<Def *>::iterator defIt = reachingDef.begin(); defIt != reachingDef.end(); defIt++) {
            Def* nodeDef = *defIt;
            if (nodeDef->getStmt() != NULL) {
                printf("\n %d: ", defCount);
                nodeDef->getStmt()->dump_tree(2);
                printf(" From Basic Block = %d", nodeDef->getBB()->getId());
            } else {
                printf("\n %d: as Arguments ", defCount);
            }
            defCount++;
        }
        printf("\n\n");
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
    mapVarToDefs globalDefsList;
    map<ast_node*, Use*> mapNodeToUse;
public:
    DefUseAnalysis(CFG* c) {
        set_for_sent(true);
        set_for_id(false);
        set_separate_post_apply(true);
        currentCFG = c;
    }

    list<foreachBBPair> getForeachToBBMap() {   return foreachToBBMap;  }
    map<ast_node*, Use*> getMapNodeToUse() {    return mapNodeToUse;    }
    Use* getUseForNode(ast_node* n) {
        map<ast_node*, Use*>::iterator nodeToUseIt = mapNodeToUse.find(n);
        if (nodeToUseIt == mapNodeToUse.end()) {
            return NULL;
        }
        return (*nodeToUseIt).second;
    }

    bool addToGlobalDefsList(ast_sent* s, ast_node* n, BasicBlock* definedInBB) {
        Def* newDef = new Def(s, n, definedInBB);
        return addToGlobalDefsList(newDef);
    }

    bool addToGlobalDefsList(Def* newDef) {
        return addToMapDefList(&globalDefsList, newDef);
    }

    bool generateKillSetFor(BasicBlock* bb) {
        mapVarToDefs BBGenSet = bb->getGenSet();
        for (mapVarToDefs::iterator genSetIt = BBGenSet.begin(); genSetIt != BBGenSet.end(); genSetIt++) {
            string varName = (*genSetIt).first;
            list<Def*> varDefList = (*genSetIt).second;
            Def* lastDefOfVarInBB = varDefList.back();

            mapVarToDefs::iterator globalDefListIt = globalDefsList.find(varName);
            if (globalDefListIt == globalDefsList.end()) {
                assert(false && "Could not find the definition for the variable");
                return false;
            }
            list<Def*> globalVarDefList = (*globalDefListIt).second;
            for (list<Def*>::iterator globalDefsIt = globalVarDefList.begin();
                globalDefsIt != globalVarDefList.end(); globalDefsIt++) {
                Def* globalVarDef = (*globalDefsIt);
                if (globalVarDef != lastDefOfVarInBB) {
                    bb->addToKillSet(globalVarDef);
                }
            }
        }
        return true;
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
                Def* newDef = new Def(NULL, id, currentBB);
                startBB->addToGenSet(newDef);
                addToGlobalDefsList(newDef);
            }
        }
    }

    bool apply(ast_sent* s) {
        if (s->get_nodetype() == AST_ASSIGN) {
            ast_assign* assignSent = (ast_assign*)s;

            if (assignSent->is_reduce_assign()){
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
                BBAfterForeach = succList.back();

                if (BBAfterForeach == NULL)
                    assert(false && "Could not find a successor of foreach loop");

                if (assignSent->is_target_scalar()) {
                    ast_id* target = assignSent->get_lhs_scala();
                    Def* newDef = new Def(s, target, BBAfterForeach);
                    BBAfterForeach->addToGenSet(newDef);
                    addToGlobalDefsList(newDef);
                } else {
                    ast_field* targetField = assignSent->get_lhs_field();
                    Def* newDef = new Def(s, targetField, BBAfterForeach);
                    BBAfterForeach->addToGenSet(newDef);
                    addToGlobalDefsList(newDef);
                }
                if (assignSent->has_lhs_list() && assignSent->is_argminmax_assign()) {
                    list<ast_node*> lhsList = assignSent->get_lhs_list();
                    list<ast_node*>::iterator lhsListIt = lhsList.begin();
                    for (; lhsListIt != lhsList.end(); lhsListIt++) {
                        ast_node* targetNode = *lhsListIt;
                        Def* newDef = new Def(s, targetNode, BBAfterForeach);
                        BBAfterForeach->addToGenSet(newDef);
                        addToGlobalDefsList(newDef);
                    }
                }
            } else { 
                Def* newDef;
                if (assignSent->is_target_scalar()) {
                    ast_node* targetNode = assignSent->get_lhs_scala();
                    newDef = new Def(s, targetNode, currentBB);
                    currentBB->addToGenSet(newDef);
                    addToGlobalDefsList(newDef);
                } else {
                    ast_node* targetNode = assignSent->get_lhs_field();
                    newDef = new Def(s, targetNode, currentBB);
                    currentBB->addToGenSet(newDef);
                    addToGlobalDefsList(newDef);
                }
            }
        } else if (s->get_nodetype() == AST_FOREACH) {
            ast_foreach* foreachSent = (ast_foreach*)s;
            Def* newDef;
            ast_node* iterNode = foreachSent->get_iterator();
            newDef = new Def(s, iterNode, currentBB);
            currentBB->addToGenSet(newDef);
            addToGlobalDefsList(newDef);
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
                Def* newDef;
                ast_node* iterNode = foreachSent->get_iterator();
                newDef = new Def(foreachSent, iterNode, currentBB);
                currentBB->addToGenSet(newDef);
                addToGlobalDefsList(newDef);
            }
        }
    }

    void analyse() {

        list<BasicBlock*> BBList = currentCFG->getBBList();
        for (list<BasicBlock*>::iterator BBListIt = BBList.begin(); BBListIt != BBList.end(); BBListIt++) {
            processBlock(*BBListIt);
        }
        for (list<BasicBlock*>::iterator BBListIt = BBList.begin(); BBListIt != BBList.end(); BBListIt++) {
            generateKillSetFor(*BBListIt);
        }
        bool hasChanged = true;
        while(hasChanged) {
            hasChanged = false;
            for (list<BasicBlock*>::iterator it = BBList.begin(); 
                    it != BBList.end(); it++) {
                BasicBlock* bb = *it;
                list<BasicBlock*>succList = bb->getSuccList();
                hasChanged |= bb->generateOut();
                for (list<BasicBlock*>::iterator succListIt = succList.begin(); 
                        succListIt != succList.end(); succListIt++) {
                    BasicBlock* succBB = *succListIt;
                    hasChanged |= bb->propogateOutToSucc(succBB);
                }
            }
        }
    }
    void print() {
        
        list<BasicBlock*> BBList = currentCFG->getBBList();
        for (list<BasicBlock*>::iterator BBListIt = BBList.begin(); BBListIt != BBList.end(); BBListIt++) {
            BasicBlock* bb = (*BBListIt);
            printf("\nBBNumber : %d\n", bb->getId());
            bb->printGenKillSet();
        }
        printf("Live IN and OUT information\n");
        for (list<BasicBlock*>::iterator BBListIt = BBList.begin(); BBListIt != BBList.end(); BBListIt++) {
            BasicBlock* bb = (*BBListIt);
            printf("\nBBNumber : %d\n", bb->getId());
            bb->printINAndOUTSets();
        }
    }
};

class globalAnalysisData{

    ast_procdef* procAST;
    CFG* currentCFG;
    DefUseAnalysis* defUse;

public:
    globalAnalysisData() {;}

    void setProcAST(ast_procdef* p) {   procAST = p;    }
    void setCFG(CFG* c) {    currentCFG = c; }
    void setDefUse(DefUseAnalysis* d) {
        defUse = d;
    }

    ast_procdef* getProcAST() { return procAST;}
    CFG* getCFG() { return currentCFG;  }
    DefUseAnalysis* getDefUse() {  return defUse;  }
};

globalAnalysisData analysisData;

void gm_cuda_opt_dependencyAnalysis::process(ast_procdef* proc) {

    analysisData.setProcAST(proc);
    analysisData.setCFG(new CFG());
    CFG* currentCFG = analysisData.getCFG();
    proc->traverse_both(currentCFG);
    currentCFG->printCFG();

    analysisData.setDefUse(new DefUseAnalysis(currentCFG));
    DefUseAnalysis* analysis = analysisData.getDefUse();
    analysis->addArgDefs(proc);
    analysis->analyse();
    analysis->print();
}

class gm_identifyBooleanReduction : public gm_apply {

    list<ast_sent*> removeStmts;
public:
    gm_identifyBooleanReduction() {
        set_for_sent(true);
        set_for_id(false);
    }

    bool apply(ast_sent* s) {
        ast_assign* assignSent = NULL;
        ast_id* targetId = NULL;
        ast_field* targetField = NULL;
        string nodeName;
        GM_REDUCE_T reduceType;
        if (s->get_nodetype() == AST_ASSIGN) {
            assignSent = (ast_assign*)s;
            reduceType = (GM_REDUCE_T)assignSent->get_reduce_type();
            if (gm_is_boolean_reduce_op(reduceType)) {
                if (assignSent->is_target_scalar()) {
                    targetId = assignSent->get_lhs_scala();
                    nodeName = targetId->get_orgname();
                } else {
                    targetField = assignSent->get_lhs_field();
                    nodeName = targetField->get_second()->get_orgname();
                }
            } else {
                return true;
            }
        } else {
            return true;
        }

        BasicBlock* enclosingBB = analysisData.getCFG()->getBBForSent(s);
        if (enclosingBB == NULL) {
            printf("OPT Bool Reduction: ");
            s->dump_tree(0);
            assert(false && "No Information about enclosing Basic Block for the sentence.\n");
            return false;
        }
        printf("Enclosing in Basic Block Number: %d\n", enclosingBB->getId());

        mapVarToDefs BBINSet = enclosingBB->getINSet();
        mapVarToDefs::iterator BBINSetIt = BBINSet.find(nodeName);
        if (BBINSetIt == BBINSet.end()) {
            //assert(false && "Could not find any definitions coming to the Basic Block\n");
            addAssignmentInstrOutsideLoop(assignSent);
            return true;
        }
        list<Def*> varDefList = (*BBINSetIt).second;
        list<Def*>::iterator varDefListIt = varDefList.begin();
        int defCount = 0;
        bool insertAssignment = true;
        for (; varDefListIt != varDefList.end(); varDefListIt++) {
            Def* reachingDef = *varDefListIt;
            if (reachingDef->getStmt() != NULL) {
                printf("\n %d: ", defCount);
                reachingDef->getStmt()->dump_tree(2);
                printf(" From Basic Block = %d", reachingDef->getBB()->getId());
                insertAssignment &= needAssignmentBeforeLoop(reachingDef, reduceType);
            } else {
                printf("\n %d: as Arguments ", defCount);
                insertAssignment &= true;
            }
            defCount++;
        }
        printf("\nNeeds Boolean Assignment: %d\n", insertAssignment);
        if (insertAssignment == true) {
            addAssignmentInstrOutsideLoop(assignSent);
        }
        replaceReductionStmt(assignSent);
    }

    bool needAssignmentBeforeLoop(Def* reachingDef, GM_REDUCE_T reduceType) {
        ast_node* node = reachingDef->getNode();
        ast_sent* s = reachingDef->getStmt();
        ast_assign* assignSent = (ast_assign*)s;
        ast_expr* rhsExpr = assignSent->get_rhs();

        if (rhsExpr->is_boolean_literal()) {
            if (reduceType == GMREDUCE_AND && rhsExpr->get_bval() == true)
                return false;
            else if (reduceType == GMREDUCE_OR && rhsExpr->get_bval() == false)
                return false;
        }
        return true;
    }

    void addAssignmentInstrOutsideLoop(ast_assign* s) {

        foreachBBPair pair;
        bool found = false;
        ast_id* boundIt = s->get_bound();
        list<foreachBBPair> foreachToBBMap = analysisData.getDefUse()->getForeachToBBMap();
        for (list<foreachBBPair>::iterator It = foreachToBBMap.begin(); It != foreachToBBMap.end(); It++) {
            pair = *It;
            if (!strcmp(pair.first->get_iterator()->get_orgname(), boundIt->get_orgname())) {
                found = true;
                break;
            }
        }
        if (!found)
            assert(false && "Bounding iterator for the reduction assignment not found");

        BasicBlock* foreachBB = pair.second, *insertIntoBB;
        printf("Basic Block of the For Loop = %d\n", foreachBB->getId());

        list<BasicBlock*> BBList = analysisData.getCFG()->getBBList();
        for (list<BasicBlock*>::iterator BBListIt = BBList.begin(); BBListIt != BBList.end(); BBListIt++) {
            BasicBlock* bb = (*BBListIt);
            if (bb->getId() == foreachBB->getId() - 1) {
                insertIntoBB = bb;
                break;
            }
        }
        ast_sent* insertAfter = (ast_sent*)insertIntoBB->getNodes().back();
        printf("\nInsert after sent: \n");
        insertAfter->dump_tree(2);

        ast_expr* rhsExpr = NULL;
        GM_REDUCE_T reduceType = (GM_REDUCE_T)s->get_reduce_type();
        if (reduceType == GMREDUCE_AND)
            rhsExpr = ast_expr::new_bval_expr(true);
        else if (reduceType == GMREDUCE_OR)
            rhsExpr = ast_expr::new_bval_expr(false);

        ast_assign* newAssignSent = NULL;
        if (s->is_target_scalar()) {
            string nodeName = s->get_lhs_scala()->get_orgname();
            ast_id* targetId = s->get_lhs_scala()->copy(true);
            newAssignSent = ast_assign::new_assign_scala(targetId, rhsExpr);
        } else {
            ast_field* sourceField = s->get_lhs_field();
            ast_id* leftId = sourceField->get_first()->copy(true);
            ast_id* rightId = sourceField->get_second()->copy(true);

            ast_field* targetField = ast_field::new_field(leftId, rightId);
            newAssignSent = ast_assign::new_assign_field(targetField, rhsExpr);
        }
        gm_add_sent_after(insertAfter, newAssignSent);
    }

    void replaceReductionStmt(ast_assign* s) {

        GM_REDUCE_T reduceType = (GM_REDUCE_T)s->get_reduce_type();
        ast_expr* newRhsExpr = NULL, *currentRhsExpr = s->get_rhs()->copy(true);
        ast_assign* newAssingSent = NULL;
        if (reduceType == GMREDUCE_AND) {
            newRhsExpr = ast_expr::new_luop_expr(currentRhsExpr->get_optype(), currentRhsExpr);
        }else {
            newRhsExpr = currentRhsExpr;
        }
        ast_expr* normalAssignRhsExpr = NULL;
        if (reduceType == GMREDUCE_AND)
            normalAssignRhsExpr = ast_expr::new_bval_expr(false);
        else if (reduceType == GMREDUCE_OR)
            normalAssignRhsExpr = ast_expr::new_bval_expr(true);

        ast_assign* normalAssignSent = NULL;
        if (s->is_target_scalar()) {
            string nodeName = s->get_lhs_scala()->get_orgname();
            ast_id* targetId = s->get_lhs_scala()->copy(true);
            normalAssignSent = ast_assign::new_assign_scala(targetId, normalAssignRhsExpr);
        } else {
            ast_field* sourceField = s->get_lhs_field();
            ast_id* leftId = sourceField->get_first()->copy(true);
            ast_id* rightId = sourceField->get_second()->copy(true);

            ast_field* targetField = ast_field::new_field(leftId, rightId);
            normalAssignSent = ast_assign::new_assign_field(targetField, normalAssignRhsExpr);
        }

        ast_if* ifStmt = ast_if::new_if(newRhsExpr, normalAssignSent, NULL);
        gm_add_sent_after(s, ifStmt);
        removeStmts.push_back(s);
    }

    void removeReductionStmts() {
        for (list<ast_sent*>::iterator sentIt = removeStmts.begin();
            sentIt != removeStmts.end(); sentIt++) {
            gm_ripoff_sent(*sentIt, true);
        }
    }
};

// For reduction statement of Boolean type, we eliminate atomics and replace with
// a normal assignment statement.
void gm_cuda_opt_removeAtomicsForBoolean::process(ast_procdef* proc) {

    gm_identifyBooleanReduction optHandler;
    proc->traverse_pre(&optHandler);
    optHandler.removeReductionStmts();
}


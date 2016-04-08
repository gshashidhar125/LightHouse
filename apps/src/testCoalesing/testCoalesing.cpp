#include <iostream>
#include <stdlib.h>
using namespace std;

void generateCompleteGraph(int numNodes) {

    for (int i = 0; i < numNodes; i++) {
        cout << i << " * " << i % 2 << " 7\n" ;
    }

    for (int i = 0; i < numNodes; i++) {
        for (int j = 0; j < numNodes; j++) {
            cout << i << " " << j << "\n";
        }
    }
}

int main(int argc, char* argv[]) {

    if (argc != 3) {
        cout << "Usage: Need 2 arguments.\n 1. Testcase type: 1(Complete Graph), 2(Random graph)\n 2. Number of nodes\n";
        exit(1);
    }
    int testcaseType, numNodes;
    
    testcaseType = atoi(argv[1]);
    numNodes = atoi(argv[2]);
    //cout << "Testcase Type = " << testcaseType << "\n";
    //cout << "NumNodes = " << numNodes << "\n";
    cout << numNodes << " " << numNodes * numNodes << "\n";

    switch(testcaseType) {
        case 1: generateCompleteGraph(numNodes);
                break;
    }
    return 0;
}

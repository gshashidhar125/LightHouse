#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace std;
int main(int argc, char* argv[]) {
    ifstream inputFile;
    inputFile.open(argv[1]);
    if (!inputFile.is_open()){
        printf("invalid file");
        exit(1);
    }
    int NumNodes, NumEdges;

    inputFile >> NumNodes >> NumEdges;
 
    int* edgeProp = new int [NumEdges + 1]();
    
    int i, j, k;
    int l;

    //For Conductance
    std::string str;
    for (i = 0; i < NumNodes; i++) {

        inputFile >> j >> str >> l;
        edgeProp[j] = l;
    }

    // For Conductance
    printf("Edge Property:\n");
    for (i = 0; i < NumNodes; i++) {
        printf("%d - %d", i, edgeProp[i]);
    }
    return 1;
}

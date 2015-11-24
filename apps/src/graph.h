#include <time.h>
#include <fstream>

using namespace std;
void printGraph(int* h_G[2]) {

    printf("Node Array:\n");
    for (int i = 0; i <= NumNodes + 1; i++) {
        printf("[%d]=%d ", i, h_G[0][i]);
    }
    printf("\n Edge Array:\n");
    for (int i = 0; i <= NumEdges; i++) {
        printf("%d ", h_G[1][i]);
    }
    printf("\n");
    /*for (int i = 0; i <= NumNodes; i++) {
        printf("%d: ", i);
        for (int j = h_G[0][i]; j < h_G[0][i + 1]; j++){
            printf("%d ", h_G[1][j]);
        }
        printf("\n");
    }*/
}
int populate(char *fileName, int* row[2]) {

    ifstream inputFile;
    inputFile.open(fileName);
    if (!inputFile.is_open()){
        printf("invalid file");
        exit(1);
    }

    inputFile >> NumNodes >> NumEdges;
 
    row[0] = new int [NumNodes + 2]();
    row[1] = new int [NumEdges + 1]();
    
    // for v_cover
    bool* edgeProp = new bool [NumEdges + 1]();
    
    int **AdjMatrix, i, j, k;
    AdjMatrix = new int* [NumNodes + 1]();
    for (i = 0; i <= NumNodes; i++) {
        AdjMatrix[i] = new int [NumNodes + 1]();
    }
    i = 0;
    // For v_cover
    bool l;
    while(i < NumEdges) {
        inputFile >> j >> k >> l;
        AdjMatrix[j][k] = 1;
        // For v_cover
        edgeProp[i] = l;
        i++;
    }

    int lastj = 0, currentIndex = 0;
    for (j = 0; j <= NumNodes; j++) {
        for (k = 0; k <= NumNodes; k++) {
            if (AdjMatrix[j][k] == 1) {

                while (lastj <= j || lastj == 0) {
                    if (lastj == 0) {
                        row[0][0] = currentIndex;
                        //row[0][1] = currentIndex;
                    } else {
                        row[0][lastj] = currentIndex;
                    }
                    lastj++;
                }
                row[1][currentIndex] = k;
                currentIndex ++;
            }
        }
    }

    /*i = NumEdges;
    int lastj = 0, currentIndex = 0;
    while(i > 0) {

        // For v_cover
        inputFile >> j >> k >> l;
        AdjMatrix[j][k] = 1;
        while (lastj <= j || lastj == 0) {
            if (lastj == 0) {
                row[0][0] = currentIndex;
                row[0][1] = currentIndex;
            }else {
                row[0][lastj] = currentIndex;
            }
            lastj++;
        }
//        if (AdjMatrix[k][j] != 1)
        row[1][currentIndex] = k;
        // For v_cover
        edgeProp[currentIndex] = l;
        currentIndex ++;
        i--;
    }*/
    //row[1][0] = 0;
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= NumNodes + 1) {
        row[0][lastj] = currentIndex;
        lastj++;
    }
    //row[0][lastj+1] = currentIndex;
/*    for (i = 0; i <= NumNodes + 1; i++)
        print("Vertex: %d = %d\n", i, row[0][i]);                         
 
    print("Second Array:\n");
    for (i = 0; i <= NumEdges; i++)
        print("Edges: Index: %d, Value = %d\n", i, row[1][i]);
*/
    j = 1;
    for (i = 1; i <= NumNodes; i++) {

        currentIndex = row[0][i];
        while (currentIndex < row[0][i+1]) {
//            print("%d %d\n", i, row[1][currentIndex]);
            if (AdjMatrix[i][row[1][currentIndex]] != 1 /*&&
                AdjMatrix[row[1][currentIndex]][i] != 1*/) {
                printf("\n\nGraph Do not Match at [%d][%d]. CurrentIndex = %d\n\n", i, row[1][currentIndex], currentIndex);
                break;
            }
            j++;
            currentIndex ++;
        }
    }
    for (i = 0; i <= NumNodes; i++) {

        delete[] AdjMatrix[i];
    }
    delete[] AdjMatrix;

    err = cudaMemcpy(G0, row[0], (NumNodes + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(G1, row[1], (NumEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    //err = cudaMemcpy(selectEdge, edgeProp, (NumEdges + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    //CUDA_ERR_CHECK;
    
    delete edgeProp;
    
    return 0;
}

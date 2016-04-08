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
__global__ void printGraphOnDevice(int* G0, int* G1, int NumNodes, int NumEdges) {

    for (int i = 0; i <= NumNodes; i++) {
        printf("%d: ", i);
        for (int j = G0[i]; j < G0[i + 1]; j++){
            printf("%d ", G1[j]);
        }
        printf("\n");
    }
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
    //bool* edgeProp = new bool [NumEdges + 1]();
    // for SSSP
    //int* edgeProp = new int [NumEdges + 1]();
    // For Conductance
    int* nodeProp = new int[NumNodes + 1];
    int* parent = new int[NumEdges + 1]();
    
    int i, j, k;
    // For v_cover
    //bool l;
    //For sssp, Conductance
    int l;

    //For Conductance
    std::string str;
    for (i = 0; i < NumNodes; i++) {

        inputFile >> j >> str >> l;
        //printf("Node : %d, Prop = %d", i, l);
        nodeProp[j] = l;
    }

    // For Conductance
    /*printf("Edge Property:\n");
    for (i = 0; i < NumNodes; i++) {
        printf("%d - %d", i, edgeProp[i]);
    }*/

    i = NumEdges;
    int lastj = 0, currentIndex = 0;
    while(i > 0) {

        // For v_cover, SSSP
        inputFile >> j >> k;
        while (lastj <= j || lastj == 0) {
            if (lastj == 0) {
                row[0][0] = currentIndex;
                row[0][1] = currentIndex;
            }else {
                row[0][lastj] = currentIndex;
            }
            lastj++;
        }
        row[1][currentIndex] = k;
        parent[currentIndex] = j;
        // For v_cover
        // For SSSP
        //edgeProp[currentIndex] = l;
        currentIndex ++;
        i--;
    }
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= NumNodes + 1) {
        row[0][lastj] = currentIndex;
        lastj++;
    }
    /*for (i = 0; i <= NumNodes + 1; i++)
        printf("Vertex: %d = %d\n", i, row[0][i]);
 
    printf("Second Array:\n");
    for (i = 0; i <= NumEdges; i++)
        printf("Edges: Index: %d, Value = %d\n", i, row[1][i]);*/

    err = cudaMemcpy(G0, row[0], (NumNodes + 2) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    err = cudaMemcpy(G1, row[1], (NumEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    // For v_cover
    //err = cudaMemcpy(selectEdge, edgeProp, (NumEdges + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    //CUDA_ERR_CHECK;
    // For SSSP
    //err = cudaMemcpy(len, edgeProp, (NumEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    //CUDA_ERR_CHECK;
    //root = 1;
    // For Conductance
    err = cudaMemcpy(member, nodeProp, (NumNodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    num = 3;

    err = cudaMemcpy(edgeFrom, parent, (NumEdges + 1) * sizeof(int), cudaMemcpyHostToDevice);
    CUDA_ERR_CHECK;
    /*printf("\n Parent Array:\n");
    for (int i = 0; i <= NumEdges; i++) {
        printf("%d ", parent[i]);
    }*/
    
    delete nodeProp;
    //printGraph(row);
    //printGraphOnDevice<<<1, 1>>>(G0, G1, NumNodes, NumEdges);
    //CUDA_ERR_CHECK;
    delete parent;
   
    return 0;
}

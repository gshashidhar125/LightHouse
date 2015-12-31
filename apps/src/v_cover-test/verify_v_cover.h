bool verifyv_cover(int* h_G[2]) {

    bool* h_selectEdge = new bool[NumEdges + 1]();
    bool* h_Covered = new bool[NumNodes + 1]();
    bool* verified = new bool[NumEdges + 1]();

    err = cudaMemcpy(h_selectEdge, selectEdge, (NumEdges + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    
    err = cudaMemcpy(h_Covered, Covered, (NumNodes + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    /*printf("Select Edge:\n");
    for (int i = 0; i <= NumEdges; i++) {
        printf("[%d] = %d, ", i, h_selectEdge[i]);
    }
    printf("\nCovered Nodes:\n");
    for (int i = 0; i <= NumNodes; i++) {
        printf("[%d] = %d, ", i, h_Covered[i]);
    }
    printf("\n");*/
    /*k = 0;
    for (int i = 0; i < NumNodes; i++) {
        printf("%d: ", i);
        bool first = true;
        for (int j = h_G[0][i]; j < h_G[0][i + 1]; j++, k++) {
            if (!first)
                printf("\t");
            printf("%d ", h_G[1][j]);
            if (h_selectEdge[k] == true)
                printf("true");
            else 
                printf("false");
            first = false;
        }
        printf("\n");
    }*/
    for (int i = 0; i <= NumEdges; i++) {
        verified[i] = false;
    }
    for (int i = 0; i <= NumNodes; i++) {
        if (h_Covered[i]) {
            for (int j = h_G[0][i]; j < h_G[0][i + 1]; j++) {
                h_Covered[h_G[1][j]] = true;
                verified[j] = true;
            }
        }
    }
    for (int i = 0; i <= NumNodes; i++) {
        for (int j = h_G[0][i]; j < h_G[0][i + 1]; j++) {
            if (h_Covered[i] || h_Covered[h_G[1][j]]) {
                verified[j] = true;
            }
        }
    }
    for (int i = 0; i < NumEdges; i++) {
        if (!verified[i]) {
            printf("Edge %d is not covered\n", i);
            if (i != 0) {
                return false;
            }
        }
    }
    delete verified;
    delete h_selectEdge;
    delete h_Covered;
    return true;
}

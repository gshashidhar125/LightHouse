#include <list>
bool verifysssp(int* h_G[2]) {

    int* h_dist = new int [NumNodes + 1]();
    int* length;// = new int [NumEdges + 1]();
    err = cudaMemcpy(h_dist, dist, (NumNodes + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;
    printf("Calculated Distances\n");
    for (int i = 0; i <= NumNodes; i++) {
        //printf("%d: %d\n", i, h_dist[i]);
        printf("%d\n", h_dist[i]);
    }
    return false;
    err = cudaMemcpy(length, len, (NumEdges + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_ERR_CHECK;

    std::list<int> processNodes;
    processNodes.push_back(root);
    int* distance = new int [NumNodes + 1]();
    for (int i = 0; i <= NumNodes; i++) {
        distance[i] = 99999;
    }
    distance[root] = 0;

    while(!processNodes.empty()) {
        for (int i = 0; i < processNodes.size(); i++) {
            int node = processNodes.front();
            processNodes.pop_front();
            for (int neighbourIndex = h_G[0][node]; neighbourIndex < h_G[0][node + 1]; neighbourIndex++) {
                int neighbour = h_G[1][neighbourIndex];
                if (distance[neighbour] > distance[node] + length[neighbourIndex]) {
                    distance[neighbour] = distance[node] + length[neighbourIndex];
                    processNodes.push_back(neighbour);
                }
            }
        }
    }
    /*printf("Verified Distances\n");
    for (int i = 0; i <= NumNodes; i++) {
        printf("%d: %d\n", i, distance[i]);
    }*/
    for (int i = 0; i <= NumNodes; i++) {
        if (h_dist[i] != distance[i]) {
            printf("Verification failed at node %d. Computed from GPU = %d, Expected = %d\n", i, h_dist[i], distance[i]);
            return false;
        }
    }

    delete length;
    delete distance;
    delete h_dist;
    return true;
}

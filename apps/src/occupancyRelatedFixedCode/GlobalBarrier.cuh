__device__ __forceinline__ void softwareBarrier() {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        gm_threadBlockBarrierReached[blockIdx.x] = 1;
        __threadfence();
        printf("BlockId = %d. Updated the arrary\n", blockIdx.x);
        printf("BlockId = %d: In = %d, Out = %d\n", blockIdx.x, gm_threadBlockBarrierReached[blockIdx.x], gm_threadBlockBarrierSignal[blockIdx.x]);
    }
//    return;

    if (blockIdx.x == 0) {

        __syncthreads();

        /*for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            //printf("SoftBarrier: Waiting for all other Blocks. Block = %d, Tid = %d\n", blockIdx.x, threadIdx.x);
            while (gm_threadBlockBarrierReached[i] == 0) {
                //printf("SoftBarrier: ThreadFenceBlock Block = %d, Tid = %d\n", blockIdx.x, threadIdx.x);
                __threadfence_block();
                //__threadfence_block();
            }
            //gm_threadBlockBarrierReached[i] = 0;
        }*/
        if (threadIdx.x == 0) {
            printf("Grid Dim = %d\n", gridDim.x);
            for (int i = 0; i < gridDim.x ; i++) {
                __threadfence();
                while (gm_threadBlockBarrierReached[i] == 0) {
                    //printf("SoftBarrier: ThreadFenceBlock Block = %d, Tid = %d\n", blockIdx.x, threadIdx.x);
                    __threadfence();
                    //__threadfence_block();
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            gm_threadBlockBarrierSignal[i] = 1;
            __threadfence();
            //printf("BlockId = %d. Reset to zero. \n", blockIdx.x);
        }
    } else {

        if (threadIdx.x == 0) {
            while (gm_threadBlockBarrierSignal[blockIdx.x] == 0) {
                //printf("SoftBarrier: Waiting to leave Barrier. Block = %d, Tid = %d\n", blockIdx.x, threadIdx.x);
                __threadfence();
                //__threadfence_block();
            }
        }
        __syncthreads();
    }
}

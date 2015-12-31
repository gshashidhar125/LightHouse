echo "soc-Epinions1.sssp.edge" > openMPTestDump
./runOpenMPVersionsListed.sh hop_dist ~/workspace/parallel-graph-algo/testcases/big/sssp/soc-Epinions1.sssp.edge "64 32 16 8 4 1" >> openMPTestDump
echo "soc-LiveJournal1.sssp.edge" >> openMPTestDump
 ./runOpenMPVersionsListed.sh hop_dist ~/workspace/parallel-graph-algo/testcases/big/sssp/soc-LiveJournal1.sssp.edge  "64 32 16 8 4 1">> openMPTestDump
echo "soc-pokec-relationships.edge" >> openMPTestDump
 ./runOpenMPVersionsListed.sh hop_dist ~/workspace/parallel-graph-algo/testcases/big/sssp/soc-pokec-relationships.sssp.edge "64 32 16 8 4 1" >> openMPTestDump
echo "com-orkut.ungraph.sssp.edge" >> openMPTestDump
 ./runOpenMPVersionsListed.sh hop_dist ~/workspace/parallel-graph-algo/testcases/big/sssp/com-orkut.ungraph.sssp.edge "64 32 16 8 4 1" >> openMPTestDump
echo "USA_ALL.txt.sssp.edge" >> openMPTestDump
./runOpenMPVersionsListed.sh hop_dist ~/workspace/parallel-graph-algo/testcases/big/sssp/USA_ALL.txt.sssp.edge "64 32 16 8 4 1" >> openMPTestDump


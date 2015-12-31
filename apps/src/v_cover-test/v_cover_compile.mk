GM_TOP= /home/shashi/workspace/green-Marl/Green-Marl
GM_GRAPH= ${GM_TOP}/apps/output_cpp/gm_graph
CC= g++
CFLAGS = -g -O3 -fopenmp -I${GM_GRAPH}/inc -I. 
LFLAGS = -L${GM_GRAPH}/lib -lgmgraph 
include ${GM_TOP}/setup.mk
include ${GM_TOP}/apps/output_cpp/common.mk

v_cover : v_cover.cc v_cover.h
	${CC} ${CFLAGS} $< ${LFLAGS} -o $@
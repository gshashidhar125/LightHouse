#! /bin/bash
process="$1"
inputGraph="$2"
res=`./$process $inputGraph output_$process_1Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=1 > OPENMP_$process_Results`
echo "1 thread result = $res"
res=`./$process $inputGraph output_$process_4Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=4 >> OPENMP_$process_Results`
echo "4 thread result = $res"
res=`./$process $inputGraph output_$process_8Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=8 >> OPENMP_$process_Results`
echo "8 thread result = $res"

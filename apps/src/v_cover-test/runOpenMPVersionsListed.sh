#! /bin/bash
process="$1"
inputGraph="$2"
threadList="$3"
graphName=$(basename $inputGraph)
echo -e "$process on OPENMP\n" > OPENMP_${process}_${graphName}_Results
for numThread in $threadList
do
    echo "Running $process with $numThread Threads" >> OPENMP_${process}_${graphName}_Results
    res=`timeout --signal=KILL 20m ./$process $inputGraph output_${process}_${numThread}Thread -GMInputFormat="edge" -GMMeasureTime=1 -GMNumThreads=$numThread >> OPENMP_${process}_${graphName}_Results`
    echo -e "$numThread threads result = $res. Status = $? \n \n" >> OPENMP_${process}_${graphName}_Results
done
#res=`./$process $inputGraph output_$process_1Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=1 > OPENMP_$process_Results`
#echo "1 thread result = $res"
#res=`./$process $inputGraph output_$process_4Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=4 >> OPENMP_$process_Results`
#echo "4 thread result = $res"
#res=`./$process $inputGraph output_$process_8Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=8 >> OPENMP_$process_Results`
#echo "8 thread result = $res"

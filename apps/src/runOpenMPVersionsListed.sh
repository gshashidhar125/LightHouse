#! /bin/bash
process="$1"
inputGraph="$2"
threadList="$3"
echo -e "$process on OPENMP\n" > OPENMP_${process}_Results
for numThread in $threadList
do
    echo "Running $process with $numThread Threads" >> OPENMP_${process}_Results
    res=`./$process $inputGraph output_${process}_${numThread}Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=$numThread >> OPENMP_${process}_Results`
    echo -e "$numThread threads result = $res \n \n" >> OPENMP_${process}_Results
done
#res=`./$process $inputGraph output_$process_1Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=1 > OPENMP_$process_Results`
#echo "1 thread result = $res"
#res=`./$process $inputGraph output_$process_4Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=4 >> OPENMP_$process_Results`
#echo "4 thread result = $res"
#res=`./$process $inputGraph output_$process_8Thread -GMInputFormat="edge" -root=1 -GMMeasureTime=1 -GMNumThreads=8 >> OPENMP_$process_Results`
#echo "8 thread result = $res"

#!/bin/bash

if [ $# != 1 ]; then 
  echo "time logfile"
fi

grep "weights sync time,"  $1 | awk '{print $11}' | awk '{sum+=$1} END {print "Weight sync Average = ", sum/NR, " iters: "NR}'
grep "gradients sync time,"  $1 | awk '{print $11}' | awk '{sum+=$1} END {print "Gradient sync Average = ", sum/NR, " iters: "NR}'
grep "forward and backword time," $1  | awk '{print $12}' | awk '{sum+=$1} END {print "FB Average = ", sum/NR, " iters: "NR}'
grep "update time," $1  | awk '{print $10}' | awk '{sum+=$1} END {print "Update average = ", sum/NR, " iters: "NR}'
grep "total time," $1  | awk '{print $10}' | awk '{sum+=$1} END {print "Total average = ", sum/NR, " iters: "NR}'


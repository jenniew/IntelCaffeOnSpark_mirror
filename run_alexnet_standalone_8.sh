#!/bin/bash

export CAFFE_ON_SPARK=/home/arda/bin/clean_CaffeOnSpark_intel
export LD_LIBRARY_PATH=${CAFFE_ON_SPARK}/caffe-public/distribute/lib:${CAFFE_ON_SPARK}/caffe-distri/distribute/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-7.0/lib64:/opt/intel/mkl/lib/intel64/:/opt/intel/lib/intel64

cd $CAFFE_ON_SPARK/models/bvlc_alexnet


~/bin/spark-1.6.0-bin-hadoop2.6/bin/spark-submit --master spark://Gondolin-Node-040:7077  \
    --driver-cores 10  \
   --driver-memory 20g  \
   --total-executor-cores 8  \
   --executor-cores 1  \
   --executor-memory 100g \
   --conf spark.locality.wait=3000000 \
   --conf spark.task.cpus=1 \
   --files ${CAFFE_ON_SPARK}/models/bvlc_alexnet/solver.prototxt,${CAFFE_ON_SPARK}/models/bvlc_alexnet/train_val.prototxt,${CAFFE_ON_SPARK}/caffe-public/data/ilsvrc12/imagenet_mean.binaryproto \
   --conf spark.driver.extraLibraryPath="${LD_LIBRARY_PATH}" \
    --conf spark.executorEnv.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    --class com.yahoo.ml.caffe.CaffeOnSpark ${CAFFE_ON_SPARK}/caffe-grid/target/caffe-grid-0.1-SNAPSHOT-jar-with-dependencies.jar  \
       -train   \
        -conf solver.prototxt  \
    -clusterSize 8 \
    -dataPartitions 8 \
    -devices 1  \
   -connection ethernet \
  -model hdfs://Gondolin-Node-016:9000/caffe/alexnet8/image_net_alexenet.model | tee /tmp/8.log 

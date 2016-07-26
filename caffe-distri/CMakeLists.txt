cmake_minimum_required(VERSION 3.5)
project(caffe_distri)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# set JAVA_HOME from environment, or default to known development locations
set(JAVA_HOME $ENV{JAVA_HOME})


#### add header files search path
include_directories(${PROJECT_SOURCE_DIR}/include)

# caffe/caffe.hpp, proto/caffe.pb.h
include_directories(${PROJECT_SOURCE_DIR}/../caffe-public/distribute/include/)

# jni.h, jni_md.h
include_directories(${JAVA_HOME}/include ${JAVA_HOME}/include/darwin)

# cuda header files
#include_directories(/usr/local/cuda/include)


set(SOURCE_FILES
    distribute/include/jni/com_yahoo_ml_jcaffe_CaffeNet.h
    distribute/include/jni/com_yahoo_ml_jcaffe_FloatArray.h
    distribute/include/jni/com_yahoo_ml_jcaffe_FloatBlob.h
    distribute/include/jni/com_yahoo_ml_jcaffe_FloatDataTransformer.h
    distribute/include/jni/com_yahoo_ml_jcaffe_Mat.h
    distribute/include/jni/com_yahoo_ml_jcaffe_MatVector.h
    distribute/include/util/threadpool/detail/future.hpp
    distribute/include/util/threadpool/detail/locking_ptr.hpp
    distribute/include/util/threadpool/detail/pool_core.hpp
    distribute/include/util/threadpool/detail/scope_guard.hpp
    distribute/include/util/threadpool/detail/worker_thread.hpp
    distribute/include/util/threadpool/future.hpp
    distribute/include/util/threadpool/pool.hpp
    distribute/include/util/threadpool/pool_adaptors.hpp
    distribute/include/util/threadpool/scheduling_policies.hpp
    distribute/include/util/threadpool/shutdown_policies.hpp
    distribute/include/util/threadpool/size_policies.hpp
    distribute/include/util/threadpool/task_adaptors.hpp
    distribute/include/util/InputAdapter.hpp
    distribute/include/util/MemoryInputAdapter.hpp
    distribute/include/util/mini_cluster.hpp
    distribute/include/util/parallel_cpu.hpp
    distribute/include/util/rdma.hpp
    distribute/include/util/rdma_sync.hpp
    distribute/include/util/socket.hpp
    distribute/include/util/socket_sync.hpp
    distribute/include/util/socket_sync_cpu.hpp
    distribute/include/util/threadpool.hpp
    distribute/include/CaffeNet.hpp
    distribute/include/common.hpp
    include/jni/com_yahoo_ml_jcaffe_CaffeNet.h
    include/jni/com_yahoo_ml_jcaffe_FloatArray.h
    include/jni/com_yahoo_ml_jcaffe_FloatBlob.h
    include/jni/com_yahoo_ml_jcaffe_FloatDataTransformer.h
    include/jni/com_yahoo_ml_jcaffe_Mat.h
    include/jni/com_yahoo_ml_jcaffe_MatVector.h
    include/util/threadpool/detail/future.hpp
    include/util/threadpool/detail/locking_ptr.hpp
    include/util/threadpool/detail/pool_core.hpp
    include/util/threadpool/detail/scope_guard.hpp
    include/util/threadpool/detail/worker_thread.hpp
    include/util/threadpool/future.hpp
    include/util/threadpool/pool.hpp
    include/util/threadpool/pool_adaptors.hpp
    include/util/threadpool/scheduling_policies.hpp
    include/util/threadpool/shutdown_policies.hpp
    include/util/threadpool/size_policies.hpp
    include/util/threadpool/task_adaptors.hpp
    include/util/InputAdapter.hpp
    include/util/MemoryInputAdapter.hpp
    include/util/mini_cluster.hpp
    include/util/parallel_cpu.hpp
    include/util/rdma.hpp
    include/util/rdma_sync.hpp
    include/util/socket.hpp
    include/util/socket_sync.hpp
    include/util/socket_sync_cpu.hpp
    include/util/threadpool.hpp
    include/CaffeNet.hpp
    include/common.hpp
    src/main/cpp/jni/JniCaffeNet.cpp
    src/main/cpp/jni/JniFloatArray.cpp
    src/main/cpp/jni/JniFloatBlob.cpp
    src/main/cpp/jni/JniFloatDataTransformer.cpp
    src/main/cpp/jni/JniMat.cpp
    src/main/cpp/jni/JniMatVector.cpp
    src/main/cpp/tools/caffe_mini_cluster.cpp
    src/main/cpp/util/blocking_queue_extra.cpp
    src/main/cpp/util/InputAdapter.cpp
    src/main/cpp/util/MemoryInputAdapter.cpp
    src/main/cpp/util/mini_cluster.cpp
    src/main/cpp/util/parallel_cpu.cpp
    src/main/cpp/util/rdma.cpp
    src/main/cpp/util/rdma_sync.cpp
    src/main/cpp/util/socket.cpp
    src/main/cpp/util/socket_sync.cpp
    src/main/cpp/util/socket_sync_cpu.cpp
    src/main/cpp/CaffeNet.cpp
    src/main/cpp/common.cpp)

add_executable(caffe_distri ${SOURCE_FILES})

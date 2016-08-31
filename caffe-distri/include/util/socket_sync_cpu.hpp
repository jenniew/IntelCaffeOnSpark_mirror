// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_SOCKET_SYNC_CPU_HPP_
#define CAFFE_DISTRI_SOCKET_SYNC_CPU_HPP_

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <vector>
#include "caffe/solver.hpp"
#include "util/parallel_cpu.hpp"
#include "util/socket.hpp"
#include "util/threadpool/threadpool.hpp"


namespace caffe {
/**
 * Synchronous data parallelism between machines over Ethernet. It builds on top
 * of the existing single node CPU code in Caffe, by adding an extra
 * step to synchronize nodes' solvers.
 *
 * During creation, the weight and gradient buffers are sharded by the number
 * of nodes in the cluster. Each node is assigned a shard for which it will
 * behave as a parameter server. All nodes contain and compute on the full
 * buffers, but are only parameter servers for a subset.
 *
 * An SGD iteration goes as follow, first each node sends its shard of weights
 * to all others. This could be implemented using a broadcast collective, but
 * since all nodes send to all others concurrently, bandwidth is uniform, and
 * point to point communication should already be optimal.
 *
 * Each node's CPU now has the weights ready. In the next step, gradients
 * are sharded, and each node sends their shards to their respective parameter
 * server peer. Transfers are again concurrent, and bandwidth uniform
 * between nodes. Each node then averages gradients for which it is parameter
 * server, and applies the solver. The solver code has not been optimized to
 * run only on the relevant shard, the remaining weights are simply ignored
 * and will be overridden during the first phase of the next iteration.
 */
template<typename Dtype>
class SocketSyncCPU : public P2PSyncCPU<Dtype> {
 public:
  SocketSyncCPU(shared_ptr<Solver<Dtype> > solver,
             const vector<shared_ptr<SocketChannel> >& peers, int rank);
  virtual ~SocketSyncCPU();
  void sync();

 protected:
  void chunk(int peer, size_t* offs, size_t* size);
  void CreateMasterBuffers(int peer);
  void CreateWorkerBuffers(int peer);

  virtual void on_start();
  virtual void on_gradients_ready();

  vector<shared_ptr<SocketChannel> > peers_;
  // Rank of the current node, MPI like
  int rank_;
  // Each node is parameter server for a shard, defined as an offset and size
  size_t own_offs_;
  size_t own_size_;
  // SocketSyncCPU weights and gradients buffers, allow send and receive
  vector<shared_ptr<SocketBuffer> > data_send_;
  vector<shared_ptr<SocketBuffer> > data_recv_;
  vector<shared_ptr<SocketBuffer> > diff_send_;
  vector<shared_ptr<SocketBuffer> > diff_recv_;

  // Weights and gradients buffers and size
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  int iter_count_;
  boost::threadpool::pool tp;
  DISABLE_COPY_AND_ASSIGN(SocketSyncCPU);
};

}  // namespace caffe

#endif

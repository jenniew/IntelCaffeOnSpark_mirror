// Copyright 2016 Yahoo Inc.
// Licensed under the terms of the Apache 2.0 license.
// Please see LICENSE file in the project root for terms.
#ifndef CAFFE_DISTRI_SOCKET_HPP_
#define CAFFE_DISTRI_SOCKET_HPP_

#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "threadpool.hpp"
#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/util/blocking_queue.hpp"


using std::vector;
using std::map;
using std::string;

namespace caffe {
class SocketChannel;
class SocketAdapter {
 public:
  volatile int port;
  explicit SocketAdapter(vector<shared_ptr<SocketChannel> > * channels);
  vector<shared_ptr<SocketChannel> > *channels;
  void start_sockt_srvr();
  string address(){
    char self_name[256];
    char port_buf[256];
    gethostname(self_name, 256);
    snprintf(port_buf, sizeof(port_buf), "%d", port);
    string address = self_name;
    address +=":";
    address += port_buf;
    return address;
  }
};

enum message_type {DIFF, DATA};
class QueuedMessage {
 public:
  int rank;
  int iter_count_;
  message_type type;
  int size;
  uint8_t* buffer;
  QueuedMessage(int _rank, int iter_count, message_type type, int size, uint8_t* buffer);
};

class SocketBuffer {
 public:
  SocketBuffer(int rank, int iterCount, message_type mt, SocketChannel* channel,
               uint8_t* buffer, size_t size, uint8_t* addr);
  uint8_t* addr() const {
    return addr_;
  }
  uint8_t* buffer() const {
    return buffer_;
  }
  const size_t size() const {
    return size_;
  }
  // Synchronously writes content to remote peer
  void Write();
  SocketBuffer* Read();
 //protected:
  SocketChannel* channel_;
  uint8_t* addr_;
  uint8_t* buffer_;
  /*const*/ size_t size_;
  int rank;
  message_type mt_;
  int iterCount_;
  string info() {
      std::stringstream sstm;
      sstm << "Iteration: " << iterCount_;
      return sstm.str();
  }
};

class SocketChannel {
 private:
  int connect_to_peer(string to_peer, string to_port);
 public:
  SocketChannel();
  ~SocketChannel();
  void Connect(string peer);
  int client_fd;
  caffe::BlockingQueue<QueuedMessage*> receive_queue;
  int serving_fd;
  int port_no;
  string peer_name;
  size_t size;
  mutable boost::mutex write_mutex_;
  string peer_info() {
      std::stringstream sstm;
      sstm << "peer_name: " << peer_name << " port_no: " << port_no
           << " client_fd: " << client_fd << " serving_fd: " << serving_fd;
      return sstm.str();
  }

  static boost::threadpool::pool tp;

  static caffe::BlockingQueue<QueuedMessage*> global_diff_receive_queue;

  static caffe::BlockingQueue<QueuedMessage*> global_data_receive_queue;

  static shared_ptr<SocketBuffer> read_next(const vector<shared_ptr<SocketBuffer> > &buffers, const message_type &mt) {
  	// Pop the message from local queue
  	QueuedMessage* qm =
  			reinterpret_cast<QueuedMessage*>(
  					(mt == DIFF ? global_diff_receive_queue : global_data_receive_queue)
  					.pop(string("trying to get message from queue")));
    LOG(INFO) << "Iteration: " << qm->iter_count_ <<  " got a message from: " << " , " << buffers[qm->rank]->channel_->peer_info();
  	shared_ptr<SocketBuffer> sb_sptr = buffers[qm->rank];
  	memcpy(sb_sptr->addr_, qm->buffer, qm->size);
  	// Free up the buffer and the wrapper object
  	delete qm->buffer;
  	delete qm;
  	return sb_sptr;
  }
};

class Socket {
 public:
    explicit Socket(const string &host, int port, bool listen);
    ~Socket();
    int descriptor() { return fd_; }

    shared_ptr<Socket> accept();
    size_t read(void *buff, size_t size);
    size_t write(void *buff, size_t size);

    uint64_t readInt() {
        // TODO loop for partial reads or writes
        uint64_t value;
        CHECK_EQ(read(&value, sizeof(uint64_t)), sizeof(uint64_t));
        return value;
    }
    void writeInt(uint64_t value) {
        CHECK_EQ(write(&value, sizeof(uint64_t)), sizeof(uint64_t));
    }
    string readStr() {
        size_t size = readInt();
        string str(size, ' ');
        CHECK_EQ(read(&str[0], size), size);
        return str;
    }
    void writeStr(const string &str) {
        writeInt(str.size());
        CHECK_EQ(write(const_cast<void*>(reinterpret_cast<const void *>
                      (str.c_str())), str.size()), str.size());
    }

 protected:
    explicit Socket(int fd) : fd_(fd) { }
    int fd_;

    DISABLE_COPY_AND_ASSIGN(Socket);
};
}  // namespace caffe

#endif

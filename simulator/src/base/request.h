#ifndef     RAMULATOR_BASE_REQUEST_H
#define     RAMULATOR_BASE_REQUEST_H

#include <vector>
#include <list>
#include <string>

#include "base/base.h"

namespace Ramulator {

struct Request { 
  Addr_t    addr = -1;
  AddrVec_t addr_vec {};
  std::string op;

  // Basic request id convention
  // 0 = Read, 1 = Write. The device spec defines all others
  // 2 = Compute TODO: might be specific to different levels
  // 3 = Subarray-level read
  // 4 = Subarray-level write
  // 5 = Bank-level read
  // 6 = Bank-level write
  struct Type {
    enum : int {
      Read = 0, 
      Write,
      Compute,
      SARead,
      SAWrite,
      BKRead,
      BKWrite
    };
  };

  int type_id = -1;    // An identifier for the type of the request
  int source_id = -1;  // An identifier for where the request is coming from (e.g., which core)

  int command = -1;          // The command that need to be issued to progress the request
  int final_command = -1;    // The final command that is needed to finish the request

  Clk_t arrive = -1;   // Clock cycle when the request arrive at the memory controller
  Clk_t depart = -1;   // Clock cycle when the request depart the memory controller

  std::function<void(Request&)> callback;

  Request(Addr_t addr, int type);
  Request(AddrVec_t addr_vec, int type);
  Request(AddrVec_t addr_vec, std::string opstr);
  Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback);
};


struct ReqBuffer {
  std::list<Request> buffer;
  size_t max_size = 32;


  using iterator = std::list<Request>::iterator;
  iterator begin() { return buffer.begin(); };
  iterator end() { return buffer.end(); };


  size_t size() const { return buffer.size(); }

  bool enqueue(const Request& request) {
    if (buffer.size() <= max_size) {
      buffer.push_back(request);
      return true;
    } else {
      return false;
    }
  }

  void remove(iterator it) {
    buffer.erase(it);
  }
};

}        // namespace Ramulator


#endif   // RAMULATOR_BASE_REQUEST_H
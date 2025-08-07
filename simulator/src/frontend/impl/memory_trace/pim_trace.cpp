#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

#include "frontend/frontend.h"
#include "base/exception.h"

#include "pim_codegen/codegen.h"

namespace Ramulator {

namespace fs = std::filesystem;

class PimTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PimTrace, "PimTrace", "PIM DRAM address vector trace.")

  private:
    std::vector<PIM::Trace> m_trace;
    std::vector<std::vector<std::vector<std::string>>> m_kernels;

    size_t m_trace_length = 0;
    size_t m_curr_trace_idx = 0;

    Logger_t m_logger;

    PIM::IPimCodeGen* m_pim_codegen;

  public:
    void init() override {
      std::string trace_path_str = param<std::string>("path").desc("Path to the load store trace file.").required();
      m_clock_ratio = param<uint>("clock_ratio").required();

      m_pim_codegen = create_child_ifce<PIM::IPimCodeGen>();

      m_logger = Logging::create_logger("PIM Trace");
      m_logger->info("Loading trace file {} ...", trace_path_str);
      init_trace(trace_path_str);
      m_logger->info("Loaded {} lines.", m_trace.size());      
    };

    void connect_memory_system(IMemorySystem* memory_system) override {
      IFrontEnd::connect_memory_system(memory_system);
      expand_trace();
    };


    void tick() override {
      if (m_curr_trace_idx > m_trace_length - 1) return;
      const PIM::Trace& t = m_trace[m_curr_trace_idx];
      bool trace_sent = m_memory_system->send({t.addr_vec, t.op});
      if (trace_sent) {
        m_curr_trace_idx = (m_curr_trace_idx + 1);
        if (m_curr_trace_idx % 100000000 == 0) {
          m_logger->info("Finished - {} / {} traces.", m_curr_trace_idx, m_trace.size());
        }
      }
    };


  private:
    void expand_trace() {
      std::vector<PIM::Trace> old_trace;
      for (auto t : m_trace) {
        old_trace.push_back(t);
      }
      m_trace.clear();
      m_trace.shrink_to_fit();
      int total_traces = 0;
      for (auto t : old_trace) {
        if (t.op != "kernel") {
          m_trace.push_back(t);
          total_traces++;
        } else {
          std::vector<PIM::Trace> kernel_trace;
          m_pim_codegen->codegen_kernel(m_kernels[t.addr_vec[0]], kernel_trace);
          m_logger->info("Kernel {}, #Insts: {}.", m_kernels[t.addr_vec[0]][0][0], kernel_trace.size());
          total_traces += kernel_trace.size();
          for (auto kt : kernel_trace) {
            m_trace.push_back(kt);
            // FIXME: set the cap to avoid process being killed
            if (m_trace.size() > 10000000) { 
              break;
            }
          }
          kernel_trace.clear();
          kernel_trace.shrink_to_fit();
        }
      }
      
      m_trace_length = m_trace.size();
      m_logger->info("After kernel expansion - {} / {} lines.", m_trace.size(), total_traces);   
    }

    void init_trace(const std::string& file_path_str) {
      fs::path trace_path(file_path_str);
      if (!fs::exists(trace_path)) {
        throw ConfigurationError("Trace {} does not exist!", file_path_str);
      }

      std::ifstream trace_file(trace_path);
      if (!trace_file.is_open()) {
        throw ConfigurationError("Trace {} cannot be opened!", file_path_str);
      }

      // Data structures for kernel trace
      std::string kernel_cmd = "";

      std::vector<std::vector<std::string>> tokens_list;

      std::string line;
      while (std::getline(trace_file, line)) {
        
        std::vector<std::string> tokens;
        tokenize(tokens, line, " ");
        if (line.empty()) {
          continue;
        }

        std::string op = ""; 
        if (tokens[0] == "R") {
          op = "read";
        } else if (tokens[0] == "W") {
          op = "write";
        } else if (tokens[0] == "C") { // Compute a row and write back the result
          op = "compute";
        } else if (tokens[0] == "SR") { // Read a row in a subarray
          op = "subarray-read";
        } else if (tokens[0] == "SW") { // Write a row back from a subarray PE
          op = "subarray-write";
        } else if (tokens[0] == "BR") { // Read a row to a bank PE
          op = "bank-read";
        } else if (tokens[0] == "BW") { // Write a row back from a bank PE
          op = "bank-write";
        } else if (tokens[0] == "conv2d" || tokens[0] == "gemm") {
          op = "kernel_start";
          kernel_cmd = tokens[0];
          tokens_list.clear();
          tokens.push_back("");
          tokens_list.push_back(tokens);
        } else if (tokens[0] == "end") {
          op = "kernel_end";
          m_kernels.push_back(tokens_list);
          kernel_cmd = "";
        } else if (kernel_cmd != "") {
          op = "kernel_desc";
          tokens_list.push_back(tokens);
        } else {
          throw ConfigurationError("Trace {} format invalid!", file_path_str);
        }

        if (op.find("kernel") == -1) {
          std::vector<std::string> addr_vec_tokens;
          tokenize(addr_vec_tokens, tokens[1], ",");

          AddrVec_t addr_vec;
          for (const auto& token : addr_vec_tokens) {
            addr_vec.push_back(std::stoll(token));
          }

          m_trace.push_back({op, addr_vec});
        } else if (op == "kernel_end") {
          op = "kernel";
          AddrVec_t addr_vec;
          addr_vec.push_back(m_kernels.size() - 1);
          m_trace.push_back({op, addr_vec});
        }
      }

      trace_file.close();
    };

    bool is_finished() override {
      return m_memory_system->finished();
    };    
};

}        // namespace Ramulator
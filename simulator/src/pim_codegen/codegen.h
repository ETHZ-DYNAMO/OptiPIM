#ifndef PIM_CODE_GEN_H
#define PIM_CODE_GEN_H

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include "pim_codegen/layout.h"

#include "memory_system/memory_system.h"
#include "base/base.h"
#include "dram/dram.h"

using namespace Ramulator;

namespace PIM {

struct Trace {
  std::string op;
  AddrVec_t addr_vec;
};

class IPimCodeGen {
  RAMULATOR_REGISTER_INTERFACE(IPimCodeGen, "PimCodeGen", "Instruction Generation based on Operator's Layout")

  public:
    std::string m_alloc_method;
    bool m_bank_buffer;
    int m_sim_timesteps;
    bool m_single_bank_opt;
    float m_timestep_scale;
    virtual void codegen(Layout* layout, std::vector<Trace>&) = 0;

    virtual void codegen_kernel(std::vector<std::vector<std::string>>, std::vector<Trace>&) = 0;

    Layout* codegen_conv2d(std::vector<std::vector<std::string>> tokens_list) 
    {
      std::vector<int> problem_dims;
      int Wdilation = 1, Hdilation = 1, Wstride = 1, Hstride = 1;
      std::vector<std::string> loops;
      std::vector<char> ptags;
      std::vector<int> dims;
      std::vector<int> banks;
      std::vector<int> rows;
      std::map<std::string, std::vector<int>> coeffs;

      for (auto tokens : tokens_list) {
        std::vector<std::string> array;
        tokenize(array, tokens[1], ",");
        if (tokens[0].find("Problem") != -1) {
          for (std::string s : array) {
            problem_dims.push_back(std::stoi(s));
          }
        }
        if (tokens[0].find("DilationStride") != -1) {
          Wdilation = std::stoi(array[0]);
          Hdilation = std::stoi(array[1]);
          Wstride = std::stoi(array[2]);
          Hstride = std::stoi(array[3]);
        }
        if (tokens[0].find("Loop") != -1) {
          for (std::string s : array) {
            loops.push_back(s);
          }
        }
        if (tokens[0].find("Bound") != -1) {
          for (std::string s : array) {
            dims.push_back(std::stoi(s));
          }
        }
        if (tokens[0].find("Tag") != -1) {
          for (std::string s : array) {
            ptags.push_back(s[0]);
          }
        }
        if (tokens[0].find("StartBankRow") != -1) {
          banks.push_back(std::stoi(array[0]));
          rows.push_back(std::stoi(array[1]));
        }
        if (tokens[0].find("Coeff") != -1) {
          std::vector<int> coeffs_vec;
          for (std::string s : array) {
            coeffs_vec.push_back(std::stoi(s));
          }
          std::vector<std::string> tag_array;
          tokenize(tag_array, tokens[0], "_");
          std::string dim_tag {tag_array[1][0]};
          coeffs[dim_tag] = coeffs_vec;
        }
      }

      ConvLayout* layout = new ConvLayout(problem_dims, Wdilation, Hdilation, Wstride, Hstride,
                         loops, ptags, dims, coeffs, banks, rows, m_dram, m_single_bank_opt);
      return layout;
    }

    Layout* codegen_gemm(std::vector<std::vector<std::string>> tokens_list) 
    {
      std::vector<int> problem_dims;
      std::vector<std::string> loops;
      std::vector<char> ptags;
      std::vector<int> dims;
      std::vector<int> banks;
      std::vector<int> rows;
      std::map<std::string, std::vector<int>> coeffs;

      for (auto tokens : tokens_list) {
        std::vector<std::string> array;
        tokenize(array, tokens[1], ",");
        if (tokens[0].find("Problem") != -1) {
          for (std::string s : array) {
            problem_dims.push_back(std::stoi(s));
          }
        }
        if (tokens[0].find("Loop") != -1) {
          for (std::string s : array) {
            loops.push_back(s);
          }
        }
        if (tokens[0].find("Bound") != -1) {
          for (std::string s : array) {
            dims.push_back(std::stoi(s));
          }
        }
        if (tokens[0].find("Tag") != -1) {
          for (std::string s : array) {
            ptags.push_back(s[0]);
          }
        }
        if (tokens[0].find("StartBankRow") != -1) {
          banks.push_back(std::stoi(array[0]));
          rows.push_back(std::stoi(array[1]));
        }
        if (tokens[0].find("Coeff") != -1) {
          std::vector<int> coeffs_vec;
          for (std::string s : array) {
            coeffs_vec.push_back(std::stoi(s));
          }
          std::vector<std::string> tag_array;
          tokenize(tag_array, tokens[0], "_");
          std::string dim_tag {tag_array[1][0]};
          coeffs[dim_tag] = coeffs_vec;
        }
      }

      GemmLayout* layout = new GemmLayout(problem_dims,
                         loops, ptags, dims, coeffs, banks, rows, m_dram, m_single_bank_opt);
      return layout;
    }

    IDRAM*  m_dram = nullptr;

    std::vector<AddrVec_t> bank_base_addr;

    int m_rows_per_bank;

  protected:
    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) {
      m_dram = memory_system->get_ifce<IDRAM>();
    }
};

}       // namespace PIM

#endif  // PIM_CODE_GEN_H
#ifndef RAMULATOR_DRAM_LAMBDAS_ROWOPEN_H
#define RAMULATOR_DRAM_LAMBDAS_ROWOPEN_H

#include <spdlog/spdlog.h>

namespace Ramulator {
namespace Lambdas {
namespace RowOpen {
namespace Subarray {
  template <class T>
  bool RDWR(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    switch (node->m_state)  {
      case T::m_states["Closed"]:
      case T::m_states["Opened"]: return false;
      case T::m_states["Selected"]: return true;
      default: {
        spdlog::error("[RowHit::Subarray] Invalid subarray state for an RD/WR command!");
        std::exit(-1);      
      }
    }
  }

  template <class T>
  bool PIM(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    switch (node->m_state)  {
      case T::m_states["Closed"]: return false;
      case T::m_states["Opened"]: 
      case T::m_states["Selected"]: return true;
      default: {
        spdlog::error("[RowHit::Subarray] Invalid subarray state for an PIM command!");
        std::exit(-1);      
      }
    }
  }
}       // namespace Subarray

namespace Bank {
  template <class T>
  bool RDWR(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    switch (node->m_state)  {
      case T::m_states["Closed"]: return false;
      case T::m_states["Opened"]: return true;
      default: {
        spdlog::error("[RowHit::Bank] Invalid bank state for an RD/WR command!");
        std::exit(-1);      
      }
    }
  }
}       // namespace Bank
}       // namespace RowHit
}       // namespace Lambdas
};      // namespace Ramulator

#endif  // RAMULATOR_DRAM_LAMBDAS_ROWOPEN_H
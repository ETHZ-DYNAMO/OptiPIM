#ifndef RAMULATOR_DRAM_LAMBDAS_PREQ_H
#define RAMULATOR_DRAM_LAMBDAS_PREQ_H

#include <spdlog/spdlog.h>

namespace Ramulator {
namespace Lambdas {
namespace Preq {
namespace Subarray {
template <class T>
int RequireRowSelected(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  switch (node->m_state) {
    case T::m_states["Closed"]: return T::m_commands["ACT"];
    case T::m_states["Opened"]: return T::m_commands["SASEL"];
    case T::m_states["Selected"]: {
      if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
        return cmd;
      } else {
        return T::m_commands["PRE"];
      }
    }    
    default: {
      spdlog::error("[Preq::Subarray] Invalid subarray state for a command!");
      std::exit(-1);      
    } 
  }
};

template <class T>
int RequireRowOpen(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  switch (node->m_state) {
    case T::m_states["Closed"]: return T::m_commands["ACT"];
    case T::m_states["Opened"]: 
    case T::m_states["Selected"]: {
      if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
        return cmd;
      } else {
        return T::m_commands["PRE"];
      }
    }    
    default: {
      spdlog::error("[Preq::Subarray] Invalid subarray state for a command!");
      std::exit(-1);      
    } 
  }
};

template <class T>
int RequireSubarrayClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  switch (node->m_state) {
    case T::m_states["Closed"]: return cmd;
    case T::m_states["Opened"]:
    case T::m_states["Selected"]: return T::m_commands["PRE"];
    default: {
      spdlog::error("[Preq::Subarray] Invalid subarray state for a command!");
      std::exit(-1);      
    } 
  }
};
} // namespace Subarray

namespace Bank {
template <class T>
int RequireRowOpen(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  switch (node->m_state) {
    case T::m_states["Closed"]: return T::m_commands["ACT"];
    case T::m_states["Opened"]: {
      if (node->m_row_state.find(target_id) != node->m_row_state.end()) {
        return cmd;
      } else {
        return T::m_commands["PRE"];
      }
    }    
    default: {
      spdlog::error("[Preq::Bank] Invalid bank state for an RD/WR command!");
      std::exit(-1);      
    } 
  }
};

template <class T>
int RequireBankClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  if constexpr (T::m_levels["row"] - T::m_levels["bank"] == 1) {
    switch (node->m_state) {
      case T::m_states["Closed"]: return cmd;
      case T::m_states["Opened"]: return T::m_commands["PRE"];
      default: {
        spdlog::error("[Preq::Bank] Invalid bank state for an RD/WR command!");
        std::exit(-1);      
      } 
    }
  } else if constexpr (T::m_levels["row"] - T::m_levels["bank"] == 2) { // there is a subarray level
    for (auto subarray: node->m_child_nodes) {
      if (subarray->m_state == T::m_states["Opened"] || subarray->m_state == T::m_states["Selected"]) {
        return T::m_commands["PREASA"];
      }
    }
    return cmd;
  }
};  
}       // namespace Bank

namespace Rank {
template <class T>
int RequireAllBanksClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  if constexpr (T::m_levels["bank"] - T::m_levels["rank"] == 1) {
    for (auto bank: node->m_child_nodes) {
      if (bank->m_state == T::m_states["Closed"]) {
        continue;
      } else {
        return T::m_commands["PREA"];
      }
    }
  } else if constexpr (T::m_levels["bank"] - T::m_levels["rank"] == 2) {
    for (auto bg : node->m_child_nodes) {
      for (auto bank: bg->m_child_nodes) {
        if (bank->m_state == T::m_states["Closed"]) {
          continue;
        } else {
          return T::m_commands["PREA"];
        }
      }
    }
  }
  return cmd;
};

template <class T>
int RequireSameBanksClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  bool all_banks_ready = true;
  typename T::Node* rank = node->m_parent_node;
  for (auto bg : rank->m_child_nodes) {
    for (auto bank : bg->m_child_nodes) {
      if (bank->m_node_id == target_id) {
        all_banks_ready &= (bank->m_state == T::m_states["Closed"]);
      }
    }
  }
  if (all_banks_ready) {
    return cmd;
  } else {
    return T::m_commands["PREsb"];
  }
};
}       // namespace Rank
namespace Channel {
  template <class T>
  int RequireAllBanksClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
    if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 2) {
      for (auto bg : node->m_child_nodes) {
        for (auto bank: bg->m_child_nodes) {
          if (bank->m_state == T::m_states["Closed"]) {
            continue;
          } else {
            return T::m_commands["PREA"];
          }
        }
      }
    } else if constexpr (T::m_levels["bank"] - T::m_levels["channel"] == 3) {
      for (auto pc : node->m_child_nodes) {
        for (auto bg : pc->m_child_nodes) {
          for (auto bank: bg->m_child_nodes) {
            if (bank->m_state == T::m_states["Closed"]) {
              continue;
            } else {
              return T::m_commands["PREA"];
            }
          }
        }
      }
    }
    return cmd;
  };
}       // namespace Channel
}       // namespace Preq
}       // namespace Lambdas
};      // namespace Ramulator

#endif  // RAMULATOR_DRAM_LAMBDAS_PREQ_H
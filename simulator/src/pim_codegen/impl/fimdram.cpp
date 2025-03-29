#include "fimdram.h"

using namespace Ramulator;

namespace PIM {

class FimDRAMCodeGen final : public IPimCodeGen, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IPimCodeGen, FimDRAMCodeGen, "FimDRAM", "CodeGen for FIMDRAM.")

  public:
    void init() override {
      m_alloc_method = param<std::string>("alloc_method").default_val("row_duplicate");
      m_bank_buffer = param<bool>("bank_buffer").default_val(true);
      m_sim_timesteps = param<int>("sim_timesteps").default_val(-1);
      m_single_bank_opt = param<bool>("single_bank_opt").default_val(true);
    }

    int add_bank_tag(int row_offset, int global_bank_id) {
      return (global_bank_id * m_rows_per_bank) + row_offset;
    }

    void codegen_bank_new(std::vector<Trace>& instructions, int global_bank_id, 
                      std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses,
                      bool output_tensor, bool weight_tensor, bool first_time_in_use = true) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      AddrVec_t base_addr = bank_base_addr[global_bank_id];
      int input_load_banks = 0;
      for (auto const& row : row_col_accesses) {
        int row_addr = row.first % m_rows_per_bank;
        int source_bank_id = row.first / m_rows_per_bank;
        AddrVec_t source_addr = bank_base_addr[source_bank_id];
        for (auto const& col : row.second) {
          source_addr[m_dram->m_levels("row")] = row_addr;
          source_addr[m_dram->m_levels("column")] = col.first;
          if (weight_tensor) {
            // We need to load values from bank to ALUs for weights
            // if (first_time_in_use) {
              std::string op = "bank-read";
              instructions.push_back({op, source_addr});
            // }
          } else if (output_tensor) {
            // We need to read partial sums out from the memory
            std::string op = "read";
            instructions.push_back({op, base_addr});
          } else {
            if (input_load_banks == 0) {
              input_load_banks = col.second;
            } else {
              assert(input_load_banks == col.second);
            }
          }
        }
      }

      if (!weight_tensor && !output_tensor) {
        
          // We need to write the input to register
          if (first_time_in_use) {
            bool already_print = false;
            
            for (int i = 0; i < input_load_banks; i++) {
              for (auto const& row : row_col_accesses) {
                if (!already_print) {
                  already_print = true;
                }
                for (auto const& col : row.second) {
                  
                  std::string op = "write";
                  AddrVec_t load_addr = bank_base_addr[global_bank_id+i];
                  instructions.push_back({op, load_addr});
                  
                }
              }
            }
          }
        
      }
    }

    void codegen_bank(std::vector<Trace>& instructions, int global_bank_id, 
                      std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses,
                      bool output_tensor, bool weight_tensor) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      AddrVec_t base_addr = bank_base_addr[global_bank_id];
      for (auto const& row : row_col_accesses) {
        int row_addr = row.first % m_rows_per_bank;
        int source_bank_id = row.first / m_rows_per_bank;
        AddrVec_t source_addr = bank_base_addr[source_bank_id];
        for (auto const& col : row.second) {
          source_addr[m_dram->m_levels("row")] = row_addr;
          source_addr[m_dram->m_levels("column")] = col.first;
          if ((source_bank_id == global_bank_id) || 
              (output_tensor && m_bank_buffer)) {
            std::string op = "bank-read";
            instructions.push_back({op, source_addr});
          } else {
            std::string op = "priority-read";
            instructions.push_back({op, source_addr});
            // TODO: need a write operation to the register
            op = "write";
            base_addr[m_dram->m_levels("row")] = row_addr;
            base_addr[m_dram->m_levels("column")] = col.first % n_values_per_dq;
            instructions.push_back({op, base_addr});
          }
        }
      }
      if (output_tensor) {
        for (auto const& row : row_col_accesses) {
          int row_addr = row.first % m_rows_per_bank;
          int source_bank_id = row.first / m_rows_per_bank;
          AddrVec_t source_addr = bank_base_addr[source_bank_id];
          for (auto const& col : row.second) {
            source_addr[m_dram->m_levels("row")] = row_addr;
            source_addr[m_dram->m_levels("column")] = col.first;
            if (source_bank_id == global_bank_id || m_bank_buffer) {
              std::string op = "bank-write";
              instructions.push_back({op, source_addr});
            } else {
              // TODO: need a read operation from the register
              std::string op = "read";
              base_addr[m_dram->m_levels("row")] = row_addr;
              base_addr[m_dram->m_levels("column")] = col.first % n_values_per_dq;
              instructions.push_back({op, base_addr});
              op = "priority-write";
              instructions.push_back({op, source_addr});
            }
          }
        }
      }
    }

    void get_addr_scheme8(int addr, int n_banks, int bank_offset, int& bank, int& row, int& col) {
      // |<-rank->|<-row->|<-col high->|<-bg->|<-bank->|<-chan->|<-col low->|<-offset ->|
      // col_high = 2 bit, col low = log(col) - col_high
      int col_high_bit = 2;
      int col_low_bit = int(log2(m_dram->get_level_size("column"))) - col_high_bit;
      col = addr & ((1<<col_low_bit)-1);
      addr = addr >> col_low_bit;
      bank = addr % n_banks;
      bank += bank_offset;
      addr = addr / n_banks;
      col += (addr & ((1<<col_high_bit)-1)) << col_low_bit;
      addr = addr >> col_high_bit;
      row = addr;
    }

    void codegen_no_duplicate(Layout* layout, std::vector<Trace>& instructions) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      int n_values_per_row = m_dram->m_organization.count[m_dram->m_levels("column")]
                            * n_values_per_dq;
      
      int temporal_steps = layout->m_sequential_steps;
      int n_spatial_elems = layout->m_spatial_banks * layout->m_spatial_cols;
      int cols_per_bank = int((n_spatial_elems - 1) / layout->m_spatial_banks + 1);

      // serialize all tensor address
      std::vector<int> tensor_base_addr;
      int cur_base_addr = 0;
      for (int i = 0; i < layout->n_tensors; i++) {
        tensor_base_addr.push_back(cur_base_addr);
        int n_elements = layout->m_bounds[i].get_hash(layout->m_bounds[i]);
        int tensor_size = n_elements / n_values_per_dq;
        cur_base_addr += tensor_size;
      }

      // calculate address for each tensor
      int tid_step = 1;
      if (m_sim_timesteps > 0) {
        tid_step = (temporal_steps-1) / m_sim_timesteps + 1;
        m_timestep_scale = (float)temporal_steps / (float)m_sim_timesteps;
      } else {
        m_timestep_scale = 1;
      }

      int sim_timesteps = 0;
      for (int tid = 0; tid < temporal_steps; tid+=tid_step) {
        
        sim_timesteps++;
        layout->infer_layout(tid);
        for (int i = 0; i < layout->n_tensors; i++) {
          bool output_tensor = (i == layout->m_output_tensor);
          bool weight_tensor = (i == layout->m_weight_tensor);
          int prev_bank_id = 0;
          std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses;
          for (int sid = 0; sid < n_spatial_elems; sid++) {
            
            int local_bank_id = int(sid / cols_per_bank);
            int global_bank_id = local_bank_id + layout->m_banks[0]; // from the start bank
            
            DataSpaceIdx tensor = layout->m_pim_tensors[i][sid][0];
            int tensor_hash = tensor.get_hash(layout->m_bounds[i]);
            int tensor_addr = tensor_base_addr[i] + tensor_hash / n_values_per_dq;
            int tensor_bank, tensor_row, tensor_col;
            // the address mapping is based on scheme8: https://github.com/SAITPublic/PIMSimulator
            get_addr_scheme8(tensor_addr, layout->m_spatial_banks, layout->m_banks[0], tensor_bank, tensor_row, tensor_col);

            // Offset in the bank
            int row_offset = tensor_row;
            row_offset += layout->m_rows[0];
            row_offset = add_bank_tag(row_offset, tensor_bank);
            int col_offset = tensor_col;
            if (global_bank_id != prev_bank_id || sid == n_spatial_elems - 1) {
              codegen_bank(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor);
              row_col_accesses.clear();
            } else {
              if (row_col_accesses.find(row_offset) != row_col_accesses.end()) {
                row_col_accesses[row_offset][col_offset]++;
              } else {
                row_col_accesses[row_offset][col_offset] = 1;
              }
            }
            prev_bank_id = global_bank_id;
          }
        }
      }
      std::cout << "SimTimeSteps:" << sim_timesteps << ", TotalSteps:" << temporal_steps << std::endl;
    }

    void codegen_new(Layout* layout, std::vector<Trace>& instructions) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      int n_values_per_row = m_dram->m_organization.count[m_dram->m_levels("column")]
                            * n_values_per_dq;
      
      int temporal_steps = layout->m_sequential_steps;
      int n_spatial_elems = layout->m_spatial_banks * layout->m_spatial_cols;
      
      int banks_per_channel = m_dram->m_organization.count[m_dram->m_levels("bank")]
                            * m_dram->m_organization.count[m_dram->m_levels("bankgroup")];
      
      int input_load_banks = layout->m_spatial_banks;
      if (input_load_banks > banks_per_channel) {
        input_load_banks = banks_per_channel;
      }
      // input_load_banks = 1;
      int cols_per_bank = int((n_spatial_elems - 1) / layout->m_spatial_banks + 1);
      int cur_row = 0, next_row = 0;

      int tid_step = 1;
      if (m_sim_timesteps > 0) {
        tid_step = (temporal_steps-1) / m_sim_timesteps + 1;
        m_timestep_scale = (float)temporal_steps / (float)m_sim_timesteps;
      } else {
        m_timestep_scale = 1;
      }
      // We only need to load input once
      std::vector<std::vector<std::map<int, int>>> tensor_row_idx;

      for (int i = 0; i < layout->n_tensors; i++) {
        std::vector<std::map<int, int>> col_tensor_row_idx;
        for (int j = 0; j < n_spatial_elems; j++) {
          std::map<int, int> tmp;
          col_tensor_row_idx.push_back(tmp);
        }
        tensor_row_idx.push_back(col_tensor_row_idx);
      }

      int sim_timesteps = 0;
      for (int tid = 0; tid < temporal_steps; tid+=tid_step) {
        sim_timesteps++;
        cur_row = next_row;
        std::vector<int> bank_tensor_col;
        for (int bb = 0; bb < layout->m_spatial_banks; bb++) {
          bank_tensor_col.push_back(0);
        }
        layout->infer_layout(tid);
        for (int i = 0; i < layout->n_tensors; i++) {
          bool output_tensor = (i == layout->m_output_tensor);
          bool weight_tensor = (i == layout->m_weight_tensor);
          int prev_bank_id = 0;
          std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses;
          int alloc_col = 0;
          bool first_time_in_col = false;
          for (int sid = 0; sid < n_spatial_elems; sid++) {
            
            int local_bank_id = int(sid / cols_per_bank);
            int global_bank_id = local_bank_id + layout->m_banks[0]; // from the start bank
            if (m_single_bank_opt) {
              if (local_bank_id > 0) {
                codegen_bank_new(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor, first_time_in_col);
                row_col_accesses.clear();
                break;
              }
            }

            DataSpaceIdx tensor = layout->m_pim_tensors[i][sid][0];
            int tensor_hash = tensor.get_hash(layout->m_bounds[i]);
            // Offset in the bank
            int tensor_offset = alloc_col;
            alloc_col++;
            // We allocate cols_per_bank (for spatial parallel), not the number of physical cols per bank
            int row_offset = cur_row;
            row_offset += layout->m_rows[0];
            if (row_offset + 1 > next_row) next_row = row_offset + 1;
            if (row_offset >= m_dram->m_organization.count[m_dram->m_levels("row")]) {
              row_offset = row_offset % m_dram->m_organization.count[m_dram->m_levels("row")];
              next_row = row_offset + 1;
            }
            row_offset = add_bank_tag(row_offset, global_bank_id);
            int col_offset = (tensor_offset % cols_per_bank) / n_values_per_dq;

            if (tensor_row_idx[i][sid].find(tensor_hash) == tensor_row_idx[i][sid].end()) {
              first_time_in_col = true;
              tensor_row_idx[i][sid][tensor_hash] = row_offset;
            } else {
              assert(first_time_in_col == false);
            }
            if (global_bank_id != prev_bank_id || sid == n_spatial_elems - 1) {
              codegen_bank_new(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor, first_time_in_col);
              row_col_accesses.clear();
            } else {
              if (row_col_accesses.find(row_offset) != row_col_accesses.end()) {
                row_col_accesses[row_offset][col_offset] = input_load_banks;
              } else {
                row_col_accesses[row_offset][col_offset] = input_load_banks;
              }
            }
            prev_bank_id = global_bank_id;
          }
        }
      }
      std::cout << "SimTimeSteps:" << sim_timesteps << ", TotalSteps:" << temporal_steps << std::endl;
    }

    void codegen_row_duplicate(Layout* layout, std::vector<Trace>& instructions) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      int n_values_per_row = m_dram->m_organization.count[m_dram->m_levels("column")]
                            * n_values_per_dq;
      
      int temporal_steps = layout->m_sequential_steps;
      int n_spatial_elems = layout->m_spatial_banks * layout->m_spatial_cols;
      int cols_per_bank = int((n_spatial_elems - 1) / layout->m_spatial_banks + 1);
      int cur_row = 0, next_row = 0;

      int tid_step = 1;
      if (m_sim_timesteps > 0) {
        tid_step = (temporal_steps-1) / m_sim_timesteps + 1;
        m_timestep_scale = (float)temporal_steps / (float)m_sim_timesteps;
      } else {
        m_timestep_scale = 1;
      }

      int sim_timesteps = 0;
      for (int tid = 0; tid < temporal_steps; tid+=tid_step) {
        sim_timesteps++;
        cur_row = next_row;
        std::vector<int> bank_tensor_col;
        for (int bb = 0; bb < layout->m_spatial_banks; bb++) {
          bank_tensor_col.push_back(0);
        }
        layout->infer_layout(tid);
        for (int i = 0; i < layout->n_tensors; i++) {
          bool output_tensor = (i == layout->m_output_tensor);
          bool weight_tensor = (i == layout->m_weight_tensor);
          int prev_bank_id = 0;
          std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses;
          std::unordered_map<int, int> tensor_col_pos;
          for (int sid = 0; sid < n_spatial_elems; sid++) {
            
            int local_bank_id = int(sid / cols_per_bank);
            int global_bank_id = local_bank_id + layout->m_banks[0]; // from the start bank
            if (m_single_bank_opt) {
              if (local_bank_id > 0) {
                codegen_bank_new(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor);
                row_col_accesses.clear();
                break;
              }
            }

            DataSpaceIdx tensor = layout->m_pim_tensors[i][sid][0];
            int tensor_hash = tensor.get_hash(layout->m_bounds[i]);
            if (tensor_col_pos.find(tensor_hash) == tensor_col_pos.end()) {
              tensor_col_pos[tensor_hash] = bank_tensor_col[local_bank_id];
              bank_tensor_col[local_bank_id]++;
            }
            // Offset in the bank
            int tensor_offset = tensor_col_pos[tensor_hash];
            int row_offset = cur_row + (tensor_offset / n_values_per_row);
            if (row_offset + 1 > next_row) next_row = row_offset;
            row_offset += layout->m_rows[0];
            row_offset = add_bank_tag(row_offset, global_bank_id);
            int col_offset = (tensor_offset % n_values_per_row) / n_values_per_dq;
            if (global_bank_id != prev_bank_id || sid == n_spatial_elems - 1) {
              codegen_bank_new(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor);
              row_col_accesses.clear();
            } else {
              if (row_col_accesses.find(row_offset) != row_col_accesses.end()) {
                row_col_accesses[row_offset][col_offset]++;
              } else {
                row_col_accesses[row_offset][col_offset] = 1;
              }
            }
            prev_bank_id = global_bank_id;
          }
        }
      }
      std::cout << "SimTimeSteps:" << sim_timesteps << ", TotalSteps:" << temporal_steps << std::endl;
    }

    void codegen_bank_duplicate(Layout* layout, std::vector<Trace>& instructions) {
      int n_values_per_dq = m_dram->m_organization.dq / m_dram->m_organization.m_pe_bits;
      int n_values_per_row = m_dram->m_organization.count[m_dram->m_levels("column")]
                            * n_values_per_dq;
      
      int temporal_steps = layout->m_sequential_steps;
      int n_spatial_elems = layout->m_spatial_banks * layout->m_spatial_cols;
      int cols_per_bank = int((n_spatial_elems - 1) / layout->m_spatial_banks + 1);
      
      // calculate the row for each tensor
      // tensor_row_idx[i][j]: address of the j-th row of tensor-i
      std::vector<std::vector<int>> tensor_row_idx;
      int cur_row = 0;
      for (int i = 0; i < layout->n_tensors; i++) {
        std::vector<int> row_idx = {cur_row};
        cur_row++;
        tensor_row_idx.push_back(row_idx);
      }

      int tid_step = 1;
      if (m_sim_timesteps > 0) {
        tid_step = (temporal_steps-1) / m_sim_timesteps + 1;
        m_timestep_scale = (float)temporal_steps / (float)m_sim_timesteps;
      } else {
        m_timestep_scale = 1;
      }

      int sim_timesteps = 0;
      for (int tid = 0; tid < temporal_steps; tid+=tid_step) {
        sim_timesteps++;
        layout->infer_layout(tid);
        for (int i = 0; i < layout->n_tensors; i++) {
          bool output_tensor = (i == layout->m_output_tensor);
          bool weight_tensor = (i == layout->m_weight_tensor);
          int prev_bank_id = 0;
          std::unordered_map<int, std::unordered_map<int, int>> row_col_accesses;
          for (int sid = 0; sid < n_spatial_elems; sid++) {
            
            int local_bank_id = int(sid / cols_per_bank);
            int global_bank_id = local_bank_id + layout->m_banks[0]; // from the start bank
            
            DataSpaceIdx tensor = layout->m_pim_tensors[i][sid][0];
            int tensor_hash = tensor.get_hash(layout->m_bounds[i]);
            // Offset in the bank
            int tensor_offset = layout->m_bank_tensor_idx[local_bank_id][i][tensor_hash];
            int row_offset = tensor_offset / n_values_per_row;
            if (row_offset > tensor_row_idx[i].size() -1) {
              tensor_row_idx[i].push_back(cur_row);
              cur_row++;
            }
            row_offset = tensor_row_idx[i][row_offset];
            row_offset = add_bank_tag(row_offset, global_bank_id);
            int col_offset = (tensor_offset % n_values_per_row) / n_values_per_dq;
            if (global_bank_id != prev_bank_id || sid == n_spatial_elems - 1) {
              codegen_bank(instructions, prev_bank_id, row_col_accesses, output_tensor, weight_tensor);
              row_col_accesses.clear();
            } else {
              if (row_col_accesses.find(row_offset) != row_col_accesses.end()) {
                row_col_accesses[row_offset][col_offset]++;
              } else {
                row_col_accesses[row_offset][col_offset] = 1;
              }
            }
            prev_bank_id = global_bank_id;
          }
        }
      }
      std::cout << "SimTimeSteps:" << sim_timesteps << ", TotalSteps:" << temporal_steps << std::endl;
    }

    void codegen(Layout* layout, std::vector<Trace>& instructions) {
      if (m_alloc_method == "bank_duplicate") {
        codegen_bank_duplicate(layout, instructions);
      } else if (m_alloc_method == "row_duplicate") {
        codegen_row_duplicate(layout, instructions);
      } else if (m_alloc_method == "no_duplicate") {
        codegen_no_duplicate(layout, instructions);
      } else if (m_alloc_method == "new") {
        codegen_new(layout, instructions);
      } 
      else {
        throw ConfigurationError("Non-supported allocation method {}!", m_alloc_method);
      }
    }

    void codegen_kernel(std::vector<std::vector<std::string>> tokens_list, std::vector<Trace>& instructions) {
      std::string kernel_string = tokens_list[0][0];
      Layout* kernel_layout = nullptr;
      if (kernel_string == "conv2d") {
        kernel_layout = IPimCodeGen::codegen_conv2d(tokens_list);
      } else if (kernel_string == "gemm") {
        kernel_layout = IPimCodeGen::codegen_gemm(tokens_list);
      }

      codegen(kernel_layout, instructions);

      delete kernel_layout;
    }

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      IPimCodeGen::setup(frontend, memory_system);
      // pre-store the base addresses for global bank id
      int bank_level = m_dram->m_levels("bank");
      int total_banks = 1;
      for (int i = 0; i <= bank_level; i++) {
        total_banks *= m_dram->m_organization.count[i];
      }
      AddrVec_t addr_vec(m_dram->m_levels.size(), 0);
      for (int i = 0; i < total_banks; i++) {
        int bank_id = i;
        for (int j = bank_level; j>=0; j--) {
          int id = bank_id % m_dram->m_organization.count[j];
          bank_id /= m_dram->m_organization.count[j];
          addr_vec[j] = id;
        }
        bank_base_addr.push_back(addr_vec);
      }

      m_rows_per_bank = 1;
      for (int i = m_dram->m_levels("bank")+1; i <= m_dram->m_levels("row");i++) {
        m_rows_per_bank *= m_dram->m_organization.count[i];
      }
    }
};

}       // namespace PIM
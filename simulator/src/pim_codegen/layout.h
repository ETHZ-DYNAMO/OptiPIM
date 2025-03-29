#ifndef PIM_LAYOUT_H
#define PIM_LAYOUT_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>

#include "memory_system/memory_system.h"
#include "base/base.h"
#include "dram/dram.h"
#include "dram/spec.h"
#include "data_space.h"
#include "nest_analysis.h"

using namespace Ramulator;

namespace  PIM {

    /*
    Representation of layout
    For a convlution or matrix multiplication:
    
    */

class Layout {
public:
    std::string m_kernel;
    // number of tensors
    int n_tensors;
    int m_output_tensor;
    int m_weight_tensor;
    /*
        for c0 : 0 to C0:
            for c1 : 0 to C1
                parallel_for s : 0 to S
                    parallel for k : 0 to K:
                        ....
        -> S * K columns
        -> C0 * C1 rows
        
        memory row list: <0xaaaa, 0xaaa1, ...>


        0xaaaa: [ Input0, Input1, ... ]
        0xaaa1: [ Filter0, Filter1, ...]
    */
    // loop_str: the name of ordered loop, depending on the kernel (e.g., conv2d - RSSSPQQCKN)
    std::vector<std::string> m_loops;
    // para_str: the parallel tag for each loop, P: parallel, N: no parallel
    std::vector<char> m_ptags;
    // dim_vec: the dimension of each ordered loop
    std::vector<int> m_dims;
    // coeff_vec: the coefficients for address mapping
    std::map<std::string, std::vector<int>> m_coeffs;
    // bank/row_vec: the list of rows for allocation
    std::vector<int> m_banks;
    std::vector<int> m_rows;
    // row_spatial_level: the level that splits the rows
    int m_row_spatial_level, m_row_temporal_level;
    int m_spatial_banks, m_spatial_cols, m_sequential_steps;
    // Architecture configuration of DRAM
    IDRAM* m_dram;

    // Optimization
    bool m_single_bank_opt;

    SpecDef m_tensors; // define all the tensor names, usually including "Inputs", "Weights", and "Outputs"
    SpecDef m_dim_names; // Kernel-specific single-char dimension names

    // dimension projection for each tensor
    std::map<int, projection_type> m_projections;

    // bounds of each tensor
    std::map<int, DataSpaceIdx> m_bounds;

    // Layout: 2D matrix with duplicated data (inferred by the layout) for each tensor
    // each value is the 4D data space
    // 2D matrix:
    //      Y-axis: spatially parallel
    //      X-axis: temporally sequential
    std::vector<std::vector<std::vector<DataSpaceIdx>>> m_pim_tensors;
    // Detailed data layout for each banks for each tensor
    std::vector<std::vector<std::unordered_map<int, int>>> m_bank_tensor_idx;
    std::vector<std::vector<int>> m_bank_tensor_count;
    // Layout: the size of parallel segment (uniform)
    std::map<int, int> m_pseg_len;

    

    Layout(std::string kernel, int _n_tensors, int _output_tensors, int _filter_tensors,
            std::vector<std::string> loops, std::vector<char> ptags, 
            std::vector<int> dims, std::map<std::string, std::vector<int>> coeffs,
            std::vector<int> banks, std::vector<int> rows, IDRAM* dram, bool single_bank_opt) 
        : m_kernel(kernel), n_tensors(_n_tensors), m_output_tensor(_output_tensors), m_weight_tensor(_filter_tensors),
          m_loops(loops), m_ptags(ptags), m_dims(dims), m_coeffs(coeffs),
          m_banks(banks), m_rows(rows), m_dram(dram), m_single_bank_opt(single_bank_opt)
    {
        get_spatial_hierarchy();
        if (m_spatial_banks == -1) {
            throw ConfigurationError("Illegal layout!");
        }

        // * Initialize the fine-grained data mapping
        // std::vector<std::vector<std::map<uint64_t, uint64_t>>> m_bank_tensor_idx;
        // std::vector<std::vector<int>> m_bank_tensor_count;
        for (int i = 0; i < m_spatial_banks; i++) {
            std::vector<std::unordered_map<int, int>> bank_tensor_idx;
            std::vector<int> bank_tensor_count;
            for (int j = 0; j < n_tensors; j++) {
                std::unordered_map<int, int> tensor_idx;
                bank_tensor_idx.push_back(tensor_idx);
                bank_tensor_count.push_back(0);
            }
            m_bank_tensor_idx.push_back(bank_tensor_idx);
            m_bank_tensor_count.push_back(bank_tensor_count);
        }
    }

    int skip_sequential(size_t& i) {
        int n_sequential_steps = 1;
        for (; i < m_ptags.size(); i++) {
            if (m_ptags[i] == 'P') {
                break;
            } else {
                n_sequential_steps *= m_dims[i];
            }
        }
        return n_sequential_steps;
    }

    int get_advance_spatial(size_t& i) {
        int n_spatial_elems = 1;
        for (; i < m_ptags.size(); i++) {
            if (m_ptags[i] != 'P') {
                break;
            }
            n_spatial_elems *= m_dims[i];
        }
        return n_spatial_elems;
    }

    void get_spatial_hierarchy() {
        bool legal = true;
        size_t i = 0;
        m_spatial_banks = get_advance_spatial(i);
        m_row_temporal_level = i;
        // skip the intermediate row level (all sequential loop)
        m_sequential_steps = skip_sequential(i);
        m_row_spatial_level = i;
        m_spatial_cols = get_advance_spatial(i);
        if (i < m_ptags.size()) legal = false;
        if (legal == false) {
            m_spatial_banks = -1;
            m_spatial_cols = -1;
            m_row_temporal_level = -1;
            m_row_spatial_level = -1;
        }
    }

    // Variables for layout inference
    int n_loops, n_spatial_elems;
    int cur_loop, spatial_id, base_spatial;
    loop_state_type loop_state, loop_base, dim_size, dim_cur_level;
    std::vector<int> timestep_index;

    void setup_timestep(int timestep, bool print) {
        if (print) {
            std::cout << "Infer Timestep: " << timestep << std::endl << "\t";
        }
        for (int level = m_row_spatial_level - 1; level >= m_row_temporal_level; level--) {
            timestep_index[level] = timestep % m_dims[level];
            timestep /= m_dims[level];
        }
    }

    void recursive_infer(int level) {
        // Only infer one bank
        if (m_single_bank_opt) {
            if (spatial_id >= m_spatial_cols) return;
        }
        if (level == m_loops.size()) { // The last level

            // Calculate indices for three tensors
            for (size_t i = 0; i < n_tensors; i++) {
                
                auto data_space_id = DataSpaceIdx::calc_data_space(loop_state, m_projections[i]);
                m_pim_tensors[i][spatial_id].push_back(data_space_id);
            }
        } else {
            std::string cur_dim = m_loops[level];
            loop_base[cur_dim] *= m_dims[level];
            
            int remaining_loop = dim_size[cur_dim] / loop_base[cur_dim];
            if (m_coeffs.size() > 0) {
                remaining_loop = 1;
                for (size_t i = dim_cur_level[cur_dim]+1; i < 3; i++) {
                    remaining_loop *= m_coeffs[cur_dim][i];
                }
            }
            int remaining_elems = 0;
            
            if (m_ptags[level] == 'P') { // Spatially parallel
                base_spatial *= m_dims[level];
                remaining_elems = n_spatial_elems / base_spatial;
                
            }
            
            for (size_t idx = 0; idx < m_dims[level]; idx++) {
                if (timestep_index[level] != -1 && idx != timestep_index[level]) continue;
                spatial_id += idx * remaining_elems;
                
                int coeff_delta = idx * remaining_loop;

                dim_cur_level[cur_dim]++;
                loop_state[cur_dim] += coeff_delta;
                recursive_infer(level+1);
                dim_cur_level[cur_dim]--;
                loop_state[cur_dim] -= coeff_delta;
                spatial_id -= idx * remaining_elems;
            }
            loop_base[cur_dim] /= m_dims[level];
            if (m_ptags[level] == 'P') {
                base_spatial /= m_dims[level];
            }
        }
    }

    void infer_layout(int timestep) {
        // 
        int n_samples_print = (m_sequential_steps-1) / 100000 + 1;
        bool print = timestep % n_samples_print == 0;
        print = false;
        // Initialize the recursive search variables
        cur_loop = 0;
        spatial_id = 0;
        n_loops = m_loops.size();
        base_spatial = 1;
        n_spatial_elems = 1;

        for (size_t i = 0; i < m_dim_names.size(); i++) {
            loop_state[std::string(m_dim_names(i))] = 0;
            loop_base[std::string(m_dim_names(i))] = 1;
            dim_size[std::string(m_dim_names(i))] = 1;
            dim_cur_level[std::string(m_dim_names(i))] = 0;
            
        }
        timestep_index.clear();
        timestep_index.shrink_to_fit();
        for (size_t i = 0; i < m_loops.size(); i++) {
            if (m_ptags[i] == 'P') {
                n_spatial_elems *= m_dims[i];
            }
            dim_size[m_loops[i]] *= m_dims[i];
            timestep_index.push_back(-1);
        }
        setup_timestep(timestep, print);

        // CHECK: if we need to further guarantee the memory is released (e.g., swap with empty)
        m_pim_tensors.clear();
        m_pim_tensors.shrink_to_fit();
        for (size_t i = 0; i < n_tensors; i++) {
            std::vector<std::vector<DataSpaceIdx>> tensor_dsi(n_spatial_elems);
            m_pim_tensors.push_back(tensor_dsi);
        }

        // Recursively infer the data space by emulating the whole loop (might be inefficient)
        recursive_infer(0);

        if (print) {
            std::cout << "Finish infer " << timestep << std::endl << std::flush;
        }
        // Detailed data layout
        int temporal_steps = m_pim_tensors[0][0].size();
        for (int tid = 0; tid < temporal_steps; tid++) {
            for (int sid = 0; sid < n_spatial_elems; sid++) {
                int cols_per_bank = int((n_spatial_elems - 1) / m_spatial_banks + 1);
                assert(cols_per_bank == m_spatial_cols);
                // int cols_per_bank = m_spatial_cols;
                int bank_id = int(sid / cols_per_bank);
                if (m_single_bank_opt) {
                    if (bank_id > 0) break;
                }
                for (size_t i = 0; i < n_tensors; i++) {
                    DataSpaceIdx tensor = m_pim_tensors[i][sid][tid];
                    int tensor_hash = tensor.get_hash(m_bounds[i]);
                    if (m_bank_tensor_idx[bank_id][i].find(tensor_hash) == m_bank_tensor_idx[bank_id][i].end()) {
                        m_bank_tensor_idx[bank_id][i][tensor_hash] = m_bank_tensor_count[bank_id][i];
                        m_bank_tensor_count[bank_id][i]++;
                    }
                }
            }
        }
        if (print) {
            std::cout << "Finish detailed layout " << timestep << std::endl << std::flush;
        }
    }

    void print_str() {
        // m_pim_tensors[i][spatial_id] - i-th tensor on the spatial_id element
        int temporal_steps = m_pim_tensors[0][0].size();
        for (int tid = 0; tid < temporal_steps; tid++) {
            std::cout << "Temporal-" << tid << ":" << std::endl;
            for (int sid = 0; sid < n_spatial_elems; sid++) {
                std::cout << "\tSpatial-" << sid << ":";
                
                for (size_t i = 0; i < m_tensors.size(); i++) {
                    
                    std::cout << "Tensor-" << m_tensors(i) << ": " << m_pim_tensors[i][sid][tid].get_str() 
                              << " - " << m_pim_tensors[i][sid][tid].get_hash(m_bounds[i]) << ", ";
                }
                std::cout << std::endl;
            }
        }

        // print data allocation
        for (int bank_id = 0; bank_id < m_spatial_banks; bank_id++) {
            std::cout << "Bank-" << bank_id << std::endl;
            for (size_t i = 0; i < m_tensors.size(); i++) {
                std::vector<int> ordered_tensors(m_bank_tensor_idx[bank_id][i].size());
                for (auto const& x : m_bank_tensor_idx[bank_id][i]) {
                    ordered_tensors[x.second] = x.first;
                }
                std::cout << "\t" << m_tensors(i) << ": ";
                for (auto idx : ordered_tensors) {
                    std::cout << idx << ", ";
                }
                std::cout << std::endl;
            }
        }
        
    }
    
};

class ConvLayout : public Layout {
public:
    inline static constexpr ImplDef m_tensors = {
        "Inputs", "Filters", "Outputs", 
    };
    /*
        for N:
            for K:
                for P:
                    for Q:
                        for C:
                            for R:
                                for S:
                                    output[x] = Func(input[y], filter[z])
                                    x = [N][K][P][Q]
                                    y = [N][C][R*WD+P*WS][S*HD+Q*HS]
                                    z = [C][K][R][S]
    */
    inline static constexpr ImplDef m_dim_names = {
        "N", "K", "P", "Q", "C", "R", "S",
    //  Bat, OCh, OH,  OW,  ICh, FH,  FW
    };
    std::vector<int> m_problem_dims;
    int m_Wdialation, m_Hdialation, m_Wstride, m_Hstride;

    ConvLayout(std::vector<int> problem_dims, int Wdilation, int Hdilation, int Wstride, int Hstride,
               std::vector<std::string> loops, std::vector<char> ptags, 
               std::vector<int> dims, std::map<std::string, std::vector<int>> coeffs,
               std::vector<int> banks, std::vector<int> rows, 
               IDRAM* dram,
               bool single_bank_opt) 
               : Layout("conv", m_tensors.size(), m_tensors("Outputs"), m_tensors("Filters"),
                        loops, ptags, dims, coeffs, banks, rows, dram, single_bank_opt) {
        // Set const values
        Layout::m_tensors = m_tensors;
        Layout::m_dim_names = m_dim_names;

        m_problem_dims = problem_dims;
        m_Wdialation = Wdilation;
        m_Hdialation = Hdilation;
        m_Wstride = Wstride;
        m_Hstride = Hstride;
        // max loop state for calculating bounds
        loop_state_type bound_loop;
        for (size_t i = 0; i < m_dim_names.size(); i++) {
            bound_loop[std::string(m_dim_names(i))] = problem_dims[i];
        }
        // Init input projection
        projection_type iproj = { /* 4-dim data, (N, C, R*Wdial + P*Wstrd, S*Hdial + Q*Hstrd)*/
            {{"N", 1}}, 
            {{"C", 1}}, 
            {{"R", m_Wdialation}, {"P", m_Wstride}},
            {{"S", m_Hdialation}, {"Q", m_Hstride}},
        };
        m_projections[m_tensors("Inputs")] = iproj;
        m_bounds[m_tensors("Inputs")] = DataSpaceIdx::calc_data_space(bound_loop, iproj);

        // Init filter projection
        projection_type fproj = { /* (C, K, R, S) */
            {{"C", 1}},
            {{"K", 1}},
            {{"R", 1}},
            {{"S", 1}},
        };
        m_projections[m_tensors("Filters")] = fproj;
        m_bounds[m_tensors("Filters")] = DataSpaceIdx::calc_data_space(bound_loop, fproj);

        // Init output projection
        projection_type oproj = { /* (N, K, P, Q) */
            {{"N", 1}},
            {{"K", 1}},
            {{"P", 1}},
            {{"Q", 1}},
        };
        m_projections[m_tensors("Outputs")] = oproj;
        m_bounds[m_tensors("Outputs")] = DataSpaceIdx::calc_data_space(bound_loop, oproj);
    }
    
};

class GemmLayout : public Layout {
public:
    inline static constexpr ImplDef m_tensors = {
        "Inputs1", "Inputs2", "Outputs", 
    };
    /*
        N*H*P*Q x N*H*Q*R -> N*H*P*R
    */
    inline static constexpr ImplDef m_dim_names = {
        "N", "H", "P", "Q", "R"
    //  Bat, Head/Ch, I1_H, I1_W, I2_W
    };
    std::vector<int> m_problem_dims;

    GemmLayout(std::vector<int> problem_dims,
               std::vector<std::string> loops, std::vector<char> ptags, 
               std::vector<int> dims, std::map<std::string, std::vector<int>> coeffs,
               std::vector<int> banks, std::vector<int> rows, 
               IDRAM* dram,
               bool single_bank_opt) 
               : Layout("gemm", m_tensors.size(), m_tensors("Outputs"), m_tensors("Inputs2"),
                        loops, ptags, dims, coeffs, banks, rows, dram, single_bank_opt) {
        // Set const values
        Layout::m_tensors = m_tensors;
        Layout::m_dim_names = m_dim_names;

        m_problem_dims = problem_dims;

        // max loop state for calculating bounds
        loop_state_type bound_loop;
        for (size_t i = 0; i < m_dim_names.size(); i++) {
            bound_loop[std::string(m_dim_names(i))] = problem_dims[i];
        }
        // Init input1 projection
        projection_type i1_proj = { /* 4-dim data, (N, H, P, Q) */
            {{"N", 1}}, 
            {{"H", 1}}, 
            {{"P", 1}},
            {{"Q", 1}},
        };
        m_projections[m_tensors("Inputs1")] = i1_proj;
        m_bounds[m_tensors("Inputs1")] = DataSpaceIdx::calc_data_space(bound_loop, i1_proj);

        // Init input2 projection
        projection_type i2_proj = { /* (N, H, Q, R) */
            {{"N", 1}},
            {{"H", 1}},
            {{"Q", 1}},
            {{"R", 1}},
        };
        m_projections[m_tensors("Inputs2")] = i2_proj;
        m_bounds[m_tensors("Inputs2")] = DataSpaceIdx::calc_data_space(bound_loop, i2_proj);

        // Init output projection
        projection_type oproj = { /* (N, H, P, R) */
            {{"N", 1}},
            {{"H", 1}},
            {{"P", 1}},
            {{"R", 1}},
        };
        m_projections[m_tensors("Outputs")] = oproj;
        m_bounds[m_tensors("Outputs")] = DataSpaceIdx::calc_data_space(bound_loop, oproj);
    }
    
};

}       // namespace PIM

#endif  // PIM_LAYOUT_H
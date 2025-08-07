#include "memory_system/memory_system.h"
#include "translation/translation.h"
#include "dram_controller/controller.h"
#include "addr_mapper/addr_mapper.h"
#include "dram/dram.h"
#include <iostream>
#include <cstdint>
#include <unordered_map>

namespace Ramulator {

class PimDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, PimDRAMSystem, "PimDRAM", "A PIM-enabled DRAM-based memory system.");

  protected:
    Clk_t m_clk = 0;
    IDRAM*  m_dram;
    IAddrMapper*  m_addr_mapper;
    std::vector<IDRAMController*> m_controllers;

  public:
    int s_num_read_requests = 0;
    int s_num_write_requests = 0;
    int s_num_other_requests = 0;
    int s_num_compute_requests = 0;
    int s_num_saread_requests = 0;
    int s_num_sawrite_requests = 0;
    int s_num_bkread_requests = 0;
    int s_num_bkwrite_requests = 0;

    int m_burst_len = 0;

  public:
    void init() override { 
      // Create device (a top-level node wrapping all channel nodes)
      m_dram = create_child_ifce<IDRAM>();
      m_addr_mapper = create_child_ifce<IAddrMapper>();

      int num_channels = m_dram->get_level_size("channel");   

      // Create memory controllers
      for (int i = 0; i < num_channels; i++) {
        IDRAMController* controller = create_child_ifce<IDRAMController>();
        controller->m_impl->set_id(fmt::format("Channel {}", i));
        controller->m_channel_id = i;
        m_controllers.push_back(controller);
      }

      m_clock_ratio = param<uint>("clock_ratio").required();

      // Report architecture parameters for cross-checking with analytical model
      int num_pch = m_dram->get_level_size("pseudochannel");
      int num_bg = m_dram->get_level_size("bankgroup");
      int pe_per_bg = m_dram->m_organization.m_pe_per_bankgroup;
      int pe_per_channel = pe_per_bg * num_bg * num_pch;
      int sys_bw = m_dram->m_organization.dq * num_pch; // bits per cycle
      int pe_bw =
          m_dram->m_organization.m_pe_bits * pe_per_channel; // bits per cycle
      m_burst_len = m_dram->m_timing_vals("nBL");
      std::cout << "[PimDRAMSystem] channels: " << num_channels
                << ", PEs/channel: " << pe_per_channel << ", SysBW: " << sys_bw
                << " bits/cycle, PEBW: " << pe_bw
                << " bits/cycle, burst length: " << m_burst_len << std::endl;

      register_stat(m_clk).name("memory_system_cycles");
      register_stat(s_num_read_requests).name("total_num_read_requests");
      register_stat(s_num_write_requests).name("total_num_write_requests");
      register_stat(s_num_other_requests).name("total_num_other_requests");
      register_stat(s_num_bkread_requests).name("total_num_bkread_requests");
      register_stat(s_num_bkwrite_requests).name("total_num_bkwrite_requests");
      register_stat(s_num_compute_requests).name("total_num_compute_requests");
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override { }

    bool send(Request req) override {
    //   m_addr_mapper->apply(req);
      // Get the request number from the string
      bool priority_cmd = false;
      if (req.op == "priority-read") {
        priority_cmd = true;
        req.op = "read";
      }
      if (req.op == "priority-write") {
        priority_cmd = true;
        req.op = "write";
      }
      req.type_id = m_dram->m_requests(req.op);
      
      int channel_id = req.addr_vec[0];
      bool is_success = false;
      if ((req.type_id == Request::Type::Read ||
          req.type_id == Request::Type::Write) && !priority_cmd) {
        is_success = m_controllers[channel_id]->send(req);
        // is_success = m_controllers[channel_id]->priority_send(req);
      } else {
        is_success = m_controllers[channel_id]->priority_send(req);
      }

      if (is_success) {
        switch (req.type_id) {
          case Request::Type::Read: {
            s_num_read_requests++;
            break;
          }
          case Request::Type::Write: {
            s_num_write_requests++;
            break;
          }
          case Request::Type::Compute: {
            s_num_compute_requests++;
            s_num_other_requests++;
            break;
          }
          case Request::Type::SARead: {
            s_num_saread_requests++;
            s_num_other_requests++;
            break;
          }
          case Request::Type::SAWrite: {
            s_num_sawrite_requests++;
            s_num_other_requests++;
            break;
          }
          case Request::Type::BKRead: {
            s_num_bkread_requests++;
            s_num_other_requests++;
            break;
          }
          case Request::Type::BKWrite: {
            s_num_bkwrite_requests++;
            s_num_other_requests++;
            break;
          }
          default: {
            s_num_other_requests++;
            break;
          }
        }
      }

      return is_success;
    };
    
    void tick() override {
      m_clk++;
      m_dram->tick();
      for (auto controller : m_controllers) {
        controller->tick();
      }
    };

    float get_tCK() override {
      return m_dram->m_timing_vals("tCK_ps") / 1000.0f;
    }

    // const SpecDef& get_supported_requests() override {
    //   return m_dram->m_requests;
    // };

    bool finished() override {
      bool all_clear = true;
      for (auto controller : m_controllers) {
        all_clear &= controller->clear();
      }
      return all_clear;
    }
};
  
}   // namespace 


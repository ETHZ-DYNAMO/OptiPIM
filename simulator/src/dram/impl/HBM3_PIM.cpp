#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

class HBM3_PIM : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, HBM3_PIM, "HBM3_PIM", "HBM3_PIM Device Model")

  public:
    inline static const std::map<std::string, Organization> org_presets = {
      //   name     density   DQ    Ch Pch  Bg Ba   Sa,   Ro    Co
      {"HBM3_2Gb",   {2<<10,  128,  {1, 2,  4,  4,  1<<4, 1<<9, 1<<6}}},
      {"HBM3_4Gb",   {4<<10,  128,  {1, 2,  4,  4,  1<<5, 1<<9, 1<<6}}},
      {"HBM3_8Gb",   {8<<10,  128,  {1, 2,  4,  4,  1<<6, 1<<9, 1<<6}}},
      {"HBM3_8Gb_new",   {8<<10,  256,  {1, 2,  4,  4,  1<<6, 1<<9, 1<<5}}},
    };

    inline static const std::map<std::string, std::vector<int>> timing_presets = {
      //   name       rate   nBL  nCL  nRCDRD  nRCDWR  nRP  nRAS  nRC  nWR  nRTPS  nRTPL  nCWL  nCCDS  nCCDL  nRRDS  nRRDL  nWTRS  nWTRL  nRTW  nFAW  nRFC  nRFCSB  nREFI  nREFISB  nRREFD  tCK_ps
      {"HBM3_2Gbps",  {2000,   4,   7,    7,      7,     7,   17,  19,   8,    2,     3,    2,    1,      2,     2,     3,     3,     4,    3,    15,   -1,   160,   3900,     -1,      8,   1000}},
      // TODO: Find more sources on HBM3 timings...
    };


  /************************************************
   *                Organization
   ***********************************************/   
    const int m_internal_prefetch_size = 2;

    inline static constexpr ImplDef m_levels = {
      "channel", "pseudochannel", "bankgroup", "bank", "subarray", "row", "column",    
    };


  /************************************************
   *             Requests & Commands
   ***********************************************/
    inline static constexpr ImplDef m_commands = {
      "ACT", 
      "PRE", "PREASA", "PREA",
      "RD",  "WR",  "RDA",  "WRA",
      "REFab", "REFsb",
      "RFMab", "RFMsb",
      "PIMOp", "SASEL", "SARD", "SAWR", // PIM: Subarray-level commands
      "BKRD", "BKWR", // PIM: Bank-level commands
    };

    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT",   "row"},
        {"PRE",   "subarray"},  {"PREASA", "bank"},  {"PREA",   "channel"},
        {"RD",    "column"},  {"WR",     "column"}, {"RDA",   "column"}, {"WRA",   "column"},
        {"REFab", "channel"}, {"REFsb",  "bank"},
        {"RFMab", "channel"}, {"RFMsb",  "bank"},
        {"PIMOp", "row"}, {"SASEL", "row"}, {"SARD", "column"}, {"SAWR", "column"}, // PIM: Subarray-level commands; PIMOp depends on PIM tech.
        {"BKRD", "column"}, {"BKWR", "column"}, // PIM: Bank-level commands
      }
    );

    inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (
      m_commands, {
                // open?   close?   access?  refresh?
        {"ACT",   {true,   false,   false,   false}},
        {"PRE",   {false,  true,    false,   false}},
        {"PREASA",{false,  true,    false,   false}},
        {"PREA",  {false,  true,    false,   false}},
        {"RD",    {false,  false,   true,    false}},
        {"WR",    {false,  false,   true,    false}},
        {"RDA",   {false,  true,    true,    false}},
        {"WRA",   {false,  true,    true,    false}},
        {"REFab", {false,  false,   false,   true }},
        {"REFsb", {false,  false,   false,   true }},
        {"RFMab", {false,  false,   false,   true }},
        {"RFMsb", {false,  false,   false,   true }},
        {"PIMOp", {false,  true,    false,   false}}, // PIMOp stores the result row back to memory using PRE
        {"SASEL", {false,  false,   false,   false}},
        {"SARD",  {false,  false,   true,    false}},
        {"SAWR",  {false,  false,   true,    false}},
        {"BKRD",  {false,  false,   true,    false}},
        {"BKWR",  {false,  false,   true,    false}}
      }
    );

    inline static constexpr ImplDef m_requests = {
      "read", "write", "all-bank-refresh", 
      "compute", "subarray-read", "subarray-write", "bank-read", "bank-write",
      "per-bank-refresh", "all-bank-rfm", "per-bank-rfm",
      
    };

    inline static const ImplLUT m_request_translations = LUT (
      m_requests, m_commands, {
        {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"}, {"per-bank-refresh", "REFsb"}, 
        {"all-bank-rfm", "RFMab"}, {"per-bank-rfm", "RFMsb"}, 
        {"compute", "PIMOp"}, {"subarray-read", "SARD"}, {"subarray-write", "SAWR"},
        {"bank-read", "BKRD"}, {"bank-write", "BKWR"}
      }
    );

   
  /************************************************
   *                   Timing
   ***********************************************/
    inline static constexpr ImplDef m_timings = {
      "rate", 
      "nBL", "nCL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC", "nWR", "nRTPS", "nRTPL", "nCWL",
      "nCCDS", "nCCDL",
      "nRRDS", "nRRDL",
      "nWTRS", "nWTRL",
      "nRTW",
      "nFAW",
      "nRFC", "nRFCSB", "nREFI", "nREFISB", "nRREFD",
      "tCK_ps"
    };


  /************************************************
   *                 Node States
   ***********************************************/
    inline static constexpr ImplDef m_states = {
       "Opened", "Closed", "Selected", "N/A"
    };

    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel",       "N/A"}, 
        {"pseudochannel", "N/A"}, 
        {"bankgroup",     "N/A"},
        {"bank",          "Closed"},
        {"subarray",      "Closed"},
        {"row",           "Closed"},
        {"column",        "N/A"},
      }
    );

  public:
    struct Node : public DRAMNodeBase<HBM3_PIM> {
      Node(HBM3_PIM* dram, Node* parent, int level, int id) : DRAMNodeBase<HBM3_PIM>(dram, parent, level, id) {};
    };
    std::vector<Node*> m_channels;
    
    FuncMatrix<ActionFunc_t<Node>>  m_actions;
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;


  public:
    void tick() override {
      m_clk++;
    };

    void init() override {
      RAMULATOR_DECLARE_SPECS();
      set_organization();
      set_timing_vals();

      set_actions();
      set_preqs();
      set_rowhits();
      set_rowopens();
      
      create_nodes();
    };

    void issue_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);
    };

    int get_preq_command(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->get_preq_command(command, addr_vec, m_clk);
    };

    bool check_ready(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_ready(command, addr_vec, m_clk);
    };

    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {
      int channel_id = addr_vec[m_levels["channel"]];
      return m_channels[channel_id]->check_rowbuffer_hit(command, addr_vec, m_clk);
    };

  private:
    void set_organization() {
      // Channel width
      m_channel_width = param_group("org").param<int>("channel_width").default_val(64);

      // Organization
      m_organization.count.resize(m_levels.size(), -1);

      // Load organization preset if provided
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {
        if (org_presets.count(*preset_name) > 0) {
          m_organization = org_presets.at(*preset_name);
        } else {
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Override the preset with any provided settings
      if (auto dq = param_group("org").param<int>("dq").optional()) {
        m_organization.dq = *dq;
      }

      for (int i = 0; i < m_levels.size(); i++){
        auto level_name = m_levels(i);
        if (auto sz = param_group("org").param<int>(level_name).optional()) {
          m_organization.count[i] = *sz;
        }
      }

      if (auto density = param_group("org").param<int>("density").optional()) {
        m_organization.density = *density;
      }

      // PIM-specific configurations
      m_organization.m_pe_per_bankgroup = param_group("org").param<int>("pe_per_bankgroup").default_val(2);
      m_organization.m_reg_per_pe = param_group("org").param<int>("reg_per_pe").default_val(2);
      m_organization.m_pe_bits = param_group("org").param<int>("pe_bits").default_val(16);

      // Sanity check: is the calculated channel density the same as the provided one?
      size_t _density = size_t(m_organization.count[m_levels["pseudochannel"]]) *
                        size_t(m_organization.count[m_levels["bankgroup"]]) *
                        size_t(m_organization.count[m_levels["bank"]]) *
                        size_t(m_organization.count[m_levels["subarray"]]) *
                        size_t(m_organization.count[m_levels["row"]]) *
                        size_t(m_organization.count[m_levels["column"]]) *
                        size_t(m_organization.dq);
      _density >>= 20;
      if (m_organization.density != _density) {
        throw ConfigurationError(
            "Calculated {} channel density {} Mb does not equal the provided density {} Mb!", 
            get_name(),
            _density, 
            m_organization.density
        );
      }

    };

    void set_timing_vals() {
      m_timing_vals.resize(m_timings.size(), -1);

      // Load timing preset if provided
      bool preset_provided = false;
      if (auto preset_name = param_group("timing").param<std::string>("preset").optional()) {
        if (timing_presets.count(*preset_name) > 0) {
          m_timing_vals = timing_presets.at(*preset_name);
          preset_provided = true;
        } else {
          throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", *preset_name, get_name());
        }
      }

      // Check for rate (in MT/s), and if provided, calculate and set tCK (in picosecond)
      if (auto dq = param_group("timing").param<int>("rate").optional()) {
        if (preset_provided) {
          throw ConfigurationError("Cannot change the transfer rate of {} when using a speed preset !", get_name());
        }
        m_timing_vals("rate") = *dq;
      }
      int tCK_ps = 1E6 / (m_timing_vals("rate") / 2);
      m_timing_vals("tCK_ps") = tCK_ps;

      // Refresh timings
      // tRFC table (unit is nanosecond!)
      constexpr int tRFC_TABLE[1][4] = {
      //  2Gb   4Gb   8Gb  16Gb
        { 160,  260,  350,  450},
      };

      // tRFC table (unit is nanosecond!)
      constexpr int tREFISB_TABLE[1][4] = {
      //  2Gb    4Gb    8Gb    16Gb
        { 4875,  4875,  2438,  2438},
      };

      int density_id = [](int density_Mb) -> int { 
        switch (density_Mb) {
          case 2048:  return 0;
          case 4096:  return 1;
          case 8192:  return 2;
          case 16384: return 3;
          default:    return -1;
        }
      }(m_organization.density);

      m_timing_vals("nRFC")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);
      m_timing_vals("nREFISB")  = JEDEC_rounding(tRFC_TABLE[0][density_id], tCK_ps);

      // Overwrite timing parameters with any user-provided value
      // Rate and tCK should not be overwritten
      for (int i = 1; i < m_timings.size() - 1; i++) {
        auto timing_name = std::string(m_timings(i));

        if (auto provided_timing = param_group("timing").param<int>(timing_name).optional()) {
          // Check if the user specifies in the number of cycles (e.g., nRCD)
          m_timing_vals(i) = *provided_timing;
        } else if (auto provided_timing = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) {
          // Check if the user specifies in nanoseconds (e.g., tRCD)
          m_timing_vals(i) = JEDEC_rounding(*provided_timing, tCK_ps);
        }
      }

      // Check if there is any uninitialized timings
      for (int i = 0; i < m_timing_vals.size(); i++) {
        if (m_timing_vals(i) == -1) {
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i));
        }
      }      

      // Set read latency
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");

      // Populate the timing constraints
      #define V(timing) (m_timing_vals(timing))
      populate_timingcons(this, {
          /*** Channel ***/ 
          /// 2-cycle ACT command (for row commands)
          {.level = "channel", .preceding = {"ACT"}, .following = {"ACT", "PRE", "PREA", "REFab", "REFsb", "RFMab", "RFMsb"}, .latency = 2},

          /*** Pseudo Channel (Table 3 — Array Access Timings Counted Individually Per Pseudo Channel, JESD-235C) ***/ 
          // RAS <-> RAS
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},
          /// 4-activation window restriction
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},

          /// ACT actually happens on the 2-nd cycle of ACT, so +1 cycle to nRRD
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRRDS") + 1},
          /// nRREFD is the latency between REFsb <-> REFsb to *different* banks
          {.level = "pseudochannel", .preceding = {"REFsb", "RFMsb"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRREFD")},
          /// nRREFD is the latency between REFsb <-> ACT to *different* banks. -1 as ACT happens on its 2nd cycle
          {.level = "pseudochannel", .preceding = {"REFsb", "RFMsb"}, .following = {"ACT"}, .latency = V("nRREFD") - 1},

          // CAS <-> CAS
          /// Data bus occupancy
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nBL")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nBL")},

          // CAS <-> CAS
          /// nCCDS is the minimal latency for column commands 
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDS")},
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDS")},
          /// RD <-> WR, Minimum Read to Write, Assuming tWPRE = 1 tCK                          
          {.level = "pseudochannel", .preceding = {"RD", "RDA"}, .following = {"WR", "WRA"}, .latency = V("nCL") + V("nBL") + 2 - V("nCWL")},
          /// WR <-> RD, Minimum Read after Write
          {.level = "pseudochannel", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRS")},
          /// CAS <-> PREab
          {.level = "pseudochannel", .preceding = {"RD"}, .following = {"PREA"}, .latency = V("nRTPS")},
          {.level = "pseudochannel", .preceding = {"WR"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")},          
          /// RAS <-> RAS
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDS")},          
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nFAW"), .window = 4},          
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"PREA"}, .latency = V("nRAS")},          
          {.level = "pseudochannel", .preceding = {"PREA"}, .following = {"ACT"}, .latency = V("nRP")},          
          /// RAS <-> REF
          {.level = "pseudochannel", .preceding = {"ACT"}, .following = {"REFab", "RFMab"}, .latency = V("nRC")},          
          {.level = "pseudochannel", .preceding = {"PRE", "PREA"}, .following = {"REFab", "RFMab"}, .latency = V("nRP")},          
          {.level = "pseudochannel", .preceding = {"RDA"}, .following = {"REFab", "RFMab"}, .latency = V("nRP") + V("nRTPS")},          
          {.level = "pseudochannel", .preceding = {"WRA"}, .following = {"REFab", "RFMab"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},          
          {.level = "pseudochannel", .preceding = {"REFab", "RFMab"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nRFC")},          

          /*** Same Bank Group ***/ 
          /// CAS <-> CAS
          {.level = "bankgroup", .preceding = {"RD", "RDA"}, .following = {"RD", "RDA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"WR", "WRA"}, .latency = V("nCCDL")},          
          {.level = "bankgroup", .preceding = {"WR", "WRA"}, .following = {"RD", "RDA"}, .latency = V("nCWL") + V("nBL") + V("nWTRL")},
          /// RAS <-> RAS
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRRDL")},  
          {.level = "bankgroup", .preceding = {"ACT"}, .following = {"REFsb", "RFMsb"}, .latency = V("nRRDL") + 1},  
          {.level = "bankgroup", .preceding = {"REFsb", "RFMsb"}, .following = {"ACT"}, .latency = V("nRRDL") - 1},  

          {.level = "bank", .preceding = {"RD"},  .following = {"PRE"}, .latency = V("nRTPS")},  

          /*** Bank ***/ // comment these constraints because of subarray-level parallelism
          // {.level = "bank", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"RD", "RDA", "BKRD"}, .latency = V("nRCDRD")},  
          {.level = "bank", .preceding = {"ACT"}, .following = {"WR", "WRA", "BKWR"}, .latency = V("nRCDWR")},  
          // {.level = "bank", .preceding = {"ACT"}, .following = {"PRE"}, .latency = V("nRAS")},  
          // {.level = "bank", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "bank", .preceding = {"RD", "BKRD"},  .following = {"PRE"}, .latency = V("nRTPL")},  
          {.level = "bank", .preceding = {"WR", "BKWR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  
          {.level = "bank", .preceding = {"RDA"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nRTPL") + V("nRP")},  
          {.level = "bank", .preceding = {"WRA"}, .following = {"ACT", "REFsb", "RFMsb"}, .latency = V("nCWL") + V("nBL") + V("nWR") + V("nRP")},  

          /*** Subarray ***/
          {.level = "subarray", .preceding = {"ACT"}, .following = {"ACT"}, .latency = V("nRC")},  
          {.level = "subarray", .preceding = {"ACT"}, .following = {"SARD"}, .latency = V("nRCDRD")},  
          {.level = "subarray", .preceding = {"ACT"}, .following = {"SAWR"}, .latency = V("nRCDWR")},  
          {.level = "subarray", .preceding = {"ACT"}, .following = {"PRE", "SASEL", "PIMOp"}, .latency = V("nRAS")},  
          {.level = "subarray", .preceding = {"PRE"}, .following = {"ACT"}, .latency = V("nRP")},  
          {.level = "subarray", .preceding = {"SARD"},  .following = {"PRE"}, .latency = V("nRTPL")},  
          {.level = "subarray", .preceding = {"SAWR"},  .following = {"PRE"}, .latency = V("nCWL") + V("nBL") + V("nWR")},  

          // from Ramulator-SALP: nRA=nCL/2=3, nWA=nCWL+nBL+nWR/2=12
          {.level = "subarray", .preceding = {"RD", "RDA"}, .following = {"SASEL", "ACT", "RD", "RDA"}, .latency = 3, .is_sibling = true}, 
          {.level = "subarray", .preceding = {"WR", "WRA"}, .following = {"SASEL", "ACT", "WR", "WRA"}, .latency = 12, .is_sibling = true},
        }
      );
      #undef V

    };

    void set_actions() {
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size()));

      // Channel Actions
      m_actions[m_levels["channel"]][m_commands["PREA"]] = Lambdas::Action::Channel::PREab<HBM3_PIM>;

      // Bank actions
      m_actions[m_levels["bank"]][m_commands["PREASA"]] = Lambdas::Action::Bank::PREASA<HBM3_PIM>;

      // Subarray actions
      m_actions[m_levels["subarray"]][m_commands["ACT"]] = Lambdas::Action::Subarray::ACT<HBM3_PIM>;
      m_actions[m_levels["subarray"]][m_commands["PRE"]] = Lambdas::Action::Subarray::PRE<HBM3_PIM>;
      m_actions[m_levels["subarray"]][m_commands["SASEL"]] = Lambdas::Action::Subarray::SASEL<HBM3_PIM>;
      m_actions[m_levels["subarray"]][m_commands["RDA"]] = Lambdas::Action::Subarray::PRE<HBM3_PIM>;
      m_actions[m_levels["subarray"]][m_commands["WRA"]] = Lambdas::Action::Subarray::PRE<HBM3_PIM>;
      m_actions[m_levels["subarray"]][m_commands["PIMOp"]] = Lambdas::Action::Subarray::PRE<HBM3_PIM>;
    };

    void set_preqs() {
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));

      // Channel Actions
      m_preqs[m_levels["channel"]][m_commands["REFab"]] = Lambdas::Preq::Channel::RequireAllBanksClosed<HBM3_PIM>;

      // Bank actions
      m_preqs[m_levels["bank"]][m_commands["REFsb"]] = Lambdas::Preq::Bank::RequireBankClosed<HBM3_PIM>;

      // Subarray actions
      m_preqs[m_levels["subarray"]][m_commands["RD"]] = Lambdas::Preq::Subarray::RequireRowSelected<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["WR"]] = Lambdas::Preq::Subarray::RequireRowSelected<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["SARD"]] = Lambdas::Preq::Subarray::RequireRowOpen<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["SAWR"]] = Lambdas::Preq::Subarray::RequireRowOpen<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["BKRD"]] = Lambdas::Preq::Subarray::RequireRowSelected<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["BKWR"]] = Lambdas::Preq::Subarray::RequireRowSelected<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["SASEL"]] = Lambdas::Preq::Subarray::RequireRowOpen<HBM3_PIM>;
      m_preqs[m_levels["subarray"]][m_commands["PIMOp"]] = Lambdas::Preq::Subarray::RequireRowOpen<HBM3_PIM>;
    };

    void set_rowhits() {
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowhits[m_levels["subarray"]][m_commands["RD"]] = Lambdas::RowHit::Subarray::RDWR<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["WR"]] = Lambdas::RowHit::Subarray::RDWR<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["BKRD"]] = Lambdas::RowHit::Subarray::RDWR<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["BKWR"]] = Lambdas::RowHit::Subarray::RDWR<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["SARD"]] = Lambdas::RowHit::Subarray::PIM<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["SAWR"]] = Lambdas::RowHit::Subarray::PIM<HBM3_PIM>;
      m_rowhits[m_levels["subarray"]][m_commands["PIMOp"]] = Lambdas::RowHit::Subarray::PIM<HBM3_PIM>;
    }


    void set_rowopens() {
      m_rowopens.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size()));

      m_rowopens[m_levels["subarray"]][m_commands["RD"]] = Lambdas::RowOpen::Subarray::RDWR<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["WR"]] = Lambdas::RowOpen::Subarray::RDWR<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["BKRD"]] = Lambdas::RowOpen::Subarray::RDWR<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["BKWR"]] = Lambdas::RowOpen::Subarray::RDWR<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["SARD"]] = Lambdas::RowOpen::Subarray::PIM<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["SAWR"]] = Lambdas::RowOpen::Subarray::PIM<HBM3_PIM>;
      m_rowopens[m_levels["subarray"]][m_commands["PIMOp"]] = Lambdas::RowOpen::Subarray::PIM<HBM3_PIM>;
    }


    void create_nodes() {
      int num_channels = m_organization.count[m_levels["channel"]];
      for (int i = 0; i < num_channels; i++) {
        Node* channel = new Node(this, nullptr, 0, i);
        m_channels.push_back(channel);
      }
    };
};


}        // namespace Ramulator
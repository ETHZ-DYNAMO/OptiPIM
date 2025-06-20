//===----------------------------------------------------------------------===//
//
// Implements the non-templated functions and methods to work with MILPs.
//
//===----------------------------------------------------------------------===//

#include "pimopt/Support/MILP.h"

#ifndef PIMOPT_GUROBI_NOT_INSTALLED

std::string pim::getGurobiOptStatusDesc(int status) {
  switch (status) {
  case 1:
    return "Model is loaded, but no solution information is available.";
  case 2:
    return "Model was solved to optimality (subject to tolerances), and an "
           "optimal solution is available.";
  case 3:
    return "Model was proven to be infeasible.";
  case 4:
    return "Model was proven to be either infeasible or unbounded. To obtain a "
           "more definitive conclusion, set the DualReductions parameter to 0 "
           "and reoptimize.";
  case 5:
    return "Model was proven to be unbounded.";
  case 9:
    return "Optimization terminated because the time expended exceeded the "
           "value specified in the TimeLimit parameter.";
  default:
    return "No description available for optimization status code " +
           std::to_string(status) +
           ". Check "
           "https://www.gurobi.com/documentation/current/refman/"
           "optimization_status_codes.html for more details";
  }
}


#endif // PIMOPT_GUROBI_NOT_INSTALLED
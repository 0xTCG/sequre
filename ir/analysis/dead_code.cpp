#include "dead_code.h"

namespace sequre {

using namespace codon::ir;

void countVarUsage( Value *instruction,
                    std::set<codon::ir::id_t> &whitelist ) {

  for ( auto *usedValue : instruction->getUsedValues() ) {
    auto *var = util::getVar(usedValue);
    if ( var ) whitelist.insert(var->getId());
    else countVarUsage(usedValue, whitelist);
  }
  
}

void eliminateDeadAssignments( SeriesFlow *series, std::set<codon::ir::id_t> &whitelist ) {
  auto it = series->begin();
  while ( it != series->end() ) {
    auto *assIns = cast<AssignInstr>(*it);
    if ( !assIns ) {
      ++it;
      continue;
    }

    auto *lhsVar = assIns->getLhs();
    if ( !whitelist.count(lhsVar->getId()) )
      it = series->erase(it);
    else ++it;
  }
}

void eliminateDeadCode( SeriesFlow *series ) {
  std::set<codon::ir::id_t> whitelist;
  for ( auto it = series->begin(); it != series->end(); ++it ) countVarUsage(*it, whitelist);
  eliminateDeadAssignments(series, whitelist);
}

} // namespace sequre

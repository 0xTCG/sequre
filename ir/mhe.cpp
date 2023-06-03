#include "mhe.h"
#include "helpers/bet.h"
#include "helpers/utils.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"
#include <iterator>
#include <math.h>

namespace sequre {

using namespace codon::ir;

std::pair<Value *, BETNode *> minimizeCipherMult( Module *M, Value *instruction, BET *bet ) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if ( retIns ) {
    auto ret = minimizeCipherMult(M, retIns->getValue(), bet);
    retIns->setValue(ret.first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *assIns = cast<AssignInstr>(instruction);
  if ( assIns ) {
    auto *lhs = assIns->getLhs();
    auto *rhs = assIns->getRhs();
    auto transformedInstruction = minimizeCipherMult(M, rhs, bet);
    if ( transformedInstruction.second )
      bet->addBET(lhs, transformedInstruction.second);
    assIns->setRhs(transformedInstruction.first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *callInstr = cast<CallInstr>(instruction);
  if ( callInstr ) {
    if ( isArithmeticOperation(getOperation(callInstr)) ) {
      auto *betNode  = parseArithmetic(callInstr);
      bet->expandNode(betNode);
      auto reduced   = bet->reduceAll(betNode);
      auto reordered = bet->reorderPriorities(betNode);
      auto *newValue = ( reduced || reordered ? generateExpression(M, betNode) : callInstr);
      return std::make_pair(newValue, betNode);
    } else {
      std::vector<Value *> newArgs;
      for ( auto arg = callInstr->begin(); arg < callInstr->end(); arg++ )
        newArgs.push_back(minimizeCipherMult(M, *arg, bet).first);
      callInstr->setArgs(newArgs);
      return std::make_pair(callInstr, nullptr);
    }
  }

  return std::make_pair(instruction, new BETNode(instruction));
}

void transformExpressions( Module *M, SeriesFlow *series ) {
  auto *bet = new BET();
  for ( auto it = series->begin(); it != series->end(); ++it ) minimizeCipherMult(M, *it, bet);
  eliminateDeadCode(series);
  // eliminateRedundance(M, series, bet);
}

/* IR passes */

void applyCipherPlainOptimizations( CallInstr *v ) {
  auto *M = v->getModule();
  auto *f = util::getFunc(v->getCallee());
  if ( !isCipherOptFunc(f) ) return;
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()));
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

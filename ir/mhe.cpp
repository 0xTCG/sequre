#include "mhe.h"
#include "helpers/bet.h"
#include "helpers/utils.h"
#include "analysis/consecutive_matmul.h"
#include "analysis/dead_code.h"
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

void transformExpressions( Module *M, SeriesFlow *series, Value *mpcValue ) {
  auto *bet = new BET();
  for ( auto it = series->begin(); it != series->end(); ++it ) minimizeCipherMult(M, *it, bet);
  eliminateDeadCode(series);
  // eliminateRedundance(M, series, bet);
  reorderConsecutiveMatmuls(series, mpcValue);
}

void applyCipherPlainOptimizations( CallInstr *v ) {
  auto *M = v->getModule();
  auto *f = util::getFunc(v->getCallee());
  if ( !isCipherOptFunc(f) ) return;
  assert( v->numArgs() > 0 && "Compile error: The first argument of the mhe_cipher_opt annotated function should be the MPC instance (annotated function has no args)" );

  auto *mpcValue = M->Nr<VarValue>(f->arg_front());
  auto mpcGenerics = mpcValue->getType()->getGenerics();
  assert( mpcGenerics.size() == 1 && "Compile error: The first argument of the mhe_cipher_opt annotated function should be the MPC instance with one and only one generic type" );
  assert(  isMPC(mpcValue, mpcGenerics[0]) && "Compile error: The first argument of the mhe_cipher_opt annotated function should be the MPC instance" );
  
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()), mpcValue);
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

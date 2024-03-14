#include "mhe.h"
#include "helpers/bet.h"
#include "helpers/utils.h"
#include "analysis/consecutive_matmul.h"
#include "analysis/dead_code.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/operator.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"
#include <iterator>
#include <math.h>

namespace sequre {

using namespace codon::ir;

/* Reordering optimizations */

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
      bet->addBET(lhs->getId(), transformedInstruction.second);
    assIns->setRhs(transformedInstruction.first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *callInstr = cast<CallInstr>(instruction);
  if ( callInstr ) {
    if ( isBinaryInstr(callInstr) ) {
      auto *betNode  = parseBinaryArithmetic(callInstr);
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
  if ( !hasCipherOptAttr(f) ) return;
  assert( v->numArgs() > 0 && "Compile error: The first argument of the mhe_cipher_opt annotated function should be the MPC instance (annotated function has no args)" );

  auto *mpcValue = M->Nr<VarValue>(f->arg_front());
  assert( isMPC(mpcValue) && "Compile error: The first argument of the mhe_cipher_opt annotated function should be the MPC instance" );
  
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()), mpcValue);
}

/* Encoding optimization */

void applyEncodingOptimization( CallInstr *v ) {
  auto *M = v->getModule();
  auto *f = util::getFunc(v->getCallee());
  if ( !hasEncOptAttr(f) ) return;
  assert( v->numArgs() > 0 && "Compile error: The first argument of the mhe_enc_opt annotated function should be the MPC instance (annotated function has no args)" );

  auto *mpcValue = M->Nr<VarValue>(f->arg_front());
  assert( isMPC(mpcValue) && "Compile error: The first argument of the mhe_enc_opt annotated function should be the MPC instance" );
  
  auto typedArgs = getTypedArgs(v, 1);  // Skipping MPC arg. Sequre funcs must have MPC instance as a first argument.
  auto args      = typedArgs.first;
  auto argsTypes = typedArgs.second;

  auto *bet = new BET();
  auto *bf  = cast<BodiedFunc>(f);
  auto *series = cast<SeriesFlow>(bf->getBody());
  bet->parseSeries(series);

  // TODO: Check if there is a better way to access func args within funcs scope in code below.
  auto *bfm = bf->getModule();
  std::vector<Var *> fargs;
  std::vector<Value *> argvs;
  auto it = f->arg_begin(); ++it;  // Skipping MPC arg. Sequre funcs must have MPC instance as a first argument.
  for (; it != f->arg_end(); ++it) {
    auto *var = *it;
    fargs.push_back(var);
    argvs.push_back(bfm->Nr<VarValue>(var)->getActual());
  }

  auto *betEncoding   = bet->getEncoding(M, fargs);
  auto *argsTuple     = util::makeTuple(argvs);
  auto *betInitHelper = getOrRealizeSequreOptimizationHelper(M, "bet_enc_init", {betEncoding->getType(), argsTuple->getType()}, {});
  assert(betInitHelper);

  auto *betInitCall = util::call(betInitHelper, {betEncoding, argsTuple});
  assert(betInitCall);

  auto *treeVarValue = util::makeVar(betInitCall, series, bf, true);
  assert(treeVarValue);

  auto *betOptHelper = getOrRealizeSequreOptimizationHelper(M, "bet_enc_opt", {treeVarValue->getType()}, {});
  assert(betInitHelper);

  auto *betOptCall = util::call(betOptHelper, {treeVarValue});
  assert(betOptCall);

  auto loc = series->begin(); ++loc;
  series->insert(loc, betOptCall);
}

/* Handle */

void MHEOptimizations::handle( CallInstr *v ) {
  applyCipherPlainOptimizations(v);
  applyEncodingOptimization(v);
}

} // namespace sequre

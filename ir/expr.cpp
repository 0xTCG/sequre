#include "expr.h"
#include "helpers/utils.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"


namespace sequre {

using namespace codon::ir;

void ExpressivenessTransformations::enableSecurity( CallInstr *v ) {
  auto *pf = getParentFunc();
  if ( !isSequreFunc(pf) ) return;
  
  auto *f = util::getFunc(v->getCallee());
  if ( !f ) return;
  
  bool isEq      = f->getUnmangledName() == Module::EQ_MAGIC_NAME;
  bool isGt      = f->getUnmangledName() == Module::GT_MAGIC_NAME;
  bool isLt      = f->getUnmangledName() == Module::LT_MAGIC_NAME;
  bool isAdd     = f->getUnmangledName() == Module::ADD_MAGIC_NAME;
  bool isSub     = f->getUnmangledName() == Module::SUB_MAGIC_NAME;
  bool isMul     = f->getUnmangledName() == Module::MUL_MAGIC_NAME;
  bool isMatMul  = f->getUnmangledName() == Module::MATMUL_MAGIC_NAME;
  bool isDiv     = f->getUnmangledName() == Module::TRUE_DIV_MAGIC_NAME;
  bool isPow     = f->getUnmangledName() == Module::POW_MAGIC_NAME;
  bool isGetItem = f->getUnmangledName() == Module::GETITEM_MAGIC_NAME;
  bool isSetItem = f->getUnmangledName() == Module::SETITEM_MAGIC_NAME;
  
  if ( !isEq &&
       !isGt &&
       !isLt && 
       !isAdd && 
       !isSub && 
       !isMul && 
       !isMatMul && 
       !isPow && 
       !isDiv &&
       !isGetItem &&
       !isSetItem )
    return;

  auto *M        = v->getModule();
  auto *mpc      = M->Nr<VarValue>(pf->arg_front());
  assert( isMPC(mpc) && "ERROR: The first argument of sequre function should be the MPC instance" );
  
  std::vector<Value *> args;
  std::vector<types::Type *> types;
  
  for ( auto it = v->begin(); it != v->end(); it++ ) {
    auto *arg = *it;
    args.push_back(arg);
    types.push_back(arg->getType());
  }

  bool isVoid    = isSetItem;
  auto *nodeType = isVoid ? types.front() : v->getType();
  
  bool isSqrtInv = false;
  if ( isDiv ) { // Special case where 1 / sqrt(x) is called
    auto *sqrtInstr = cast<CallInstr>(args.back());
    if ( sqrtInstr ) {
      auto *sqrtFunc = util::getFunc(sqrtInstr->getCallee());
      if ( sqrtFunc )
        isSqrtInv = sqrtFunc->getUnmangledName() == "sqrt";
    }
  }

  if ( !isSecureContainer(nodeType) ) return;
  if ( isMP(nodeType) && !isSqrtInv ) return;
  if ( isSharetensor(nodeType) ) {
    bool lhsIsInt = types.front()->is(M->getIntType());
    bool rhsIsInt = types.back()->is(M->getIntType());

    if ( isGetItem || isSetItem ) return;
    if ( isMul && lhsIsInt ) return;
    if ( isMul && rhsIsInt ) return;
    if ( isDiv && lhsIsInt && !isSqrtInv ) return;
    if ( isPow && lhsIsInt ) return;
    if ( isPow && !rhsIsInt ) return;
  }

  std::string methodName = isEq        ? "secure_eq"
                           : isGt      ? "secure_gt"
                           : isLt      ? "secure_lt"
                           : isAdd     ? "secure_add"
                           : isSub     ? "secure_sub"
                           : isMul     ? "secure_mul"
                           : isMatMul  ? "secure_matmul"
                           : isSqrtInv ? "secure_sqrt_inv"
                           : isDiv     ? "secure_div"
                           : isPow     ? "secure_pow"
                           : isGetItem ? "secure_getitem"
                           : isSetItem ? "secure_setitem"
                                       : "invalid_operation";
  if ( isSqrtInv ) {
    args.back() = cast<CallInstr>(args.back())->back();
    types.back() = args.back()->getType();
  }

  args.insert(args.begin(), mpc);
  types.insert(types.begin(), mpc->getType());

  auto *method = getOrRealizeSequreInternalMethod(M, methodName, types, {});
  if ( !method ) {
    std::cout << "Called at " << v->getSrcInfo() << std::endl;
    std::cout << "within " << pf->getName() << std::endl;
    assert(false && "Aborting due to unsuccessful method realization ...");
  }

  auto *func = util::call(method, args);
  v->replaceAll(func);
}

void ExpressivenessTransformations::handle( CallInstr *v ) { enableSecurity(v); }

} // namespace sequre

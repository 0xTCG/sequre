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
  
  if ( !isEq &&
       !isGt &&
       !isLt && 
       !isAdd && 
       !isSub && 
       !isMul && 
       !isMatMul && 
       !isPow && 
       !isDiv &&
       !isGetItem )
    return;

  auto *M        = v->getModule();
  auto *self     = M->Nr<VarValue>(pf->arg_front());
  auto *selfType = self->getType();
  auto *lhs      = v->front();
  auto *rhs      = v->back();
  auto *lhsType  = lhs->getType();
  auto *rhsType  = rhs->getType();

  bool isSqrtInv = false;
  if ( isDiv ) { // Special case where 1 / sqrt(x) is called
    auto *sqrtInstr = cast<CallInstr>(rhs);
    if ( sqrtInstr ) {
      auto *sqrtFunc = util::getFunc(sqrtInstr->getCallee());
      if ( sqrtFunc )
        isSqrtInv = sqrtFunc->getUnmangledName() == "sqrt";
    }
  }

  bool lhs_is_secure_container = isSecureContainer(lhsType);
  bool rhs_is_secure_container = isSecureContainer(rhsType);
  if ( !lhs_is_secure_container && !rhs_is_secure_container ) return;
  if ( isSharedTensor(lhsType) && isGetItem ) return;

  bool lhs_is_int = lhsType->is(M->getIntType());
  bool rhs_is_int = rhsType->is(M->getIntType());

  if ( isMul && lhs_is_int ) return;
  if ( isMul && rhs_is_int ) return;
  if ( isDiv && lhs_is_int && !isSqrtInv ) return;
  if ( isPow && lhs_is_int ) return;
  if ( isPow && !rhs_is_int ) return;

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
                                       : "invalid_operation";
  if ( isSqrtInv ) {
    rhs = cast<CallInstr>(rhs)->back();
    rhsType = rhs->getType();
  }

  auto *method = getOrRealizeSequreInternalMethod(M, methodName, {selfType, lhsType, rhsType}, {});
  if ( !method ) {
    std::cout << "Called within " << pf->getName() << std::endl;
    return;
  }

  auto *func = util::call(method, {self, lhs, rhs});
  v->replaceAll(func);
}

void ExpressivenessTransformations::handle( CallInstr *v ) { enableSecurity(v); }

} // namespace sequre

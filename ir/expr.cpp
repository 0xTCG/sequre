#include "expr.h"
#include "utils.h"
#include "codon/sir/util/cloning.h"
#include "codon/sir/util/irtools.h"
#include "codon/sir/util/matching.h"


namespace sequre {

using namespace codon::ir;

void ExpressivenessTransformations::transform(CallInstr *v) {
  auto *pf = getParentFunc();
  if (!isSequreFunc(pf))
    return;
  auto *f = util::getFunc(v->getCallee());
  if (!f)
    return;
  bool isEq = f->getName().find("__eq__") != std::string::npos;
  bool isGt = f->getName().find("__gt__") != std::string::npos;
  bool isLt = f->getName().find("__lt__") != std::string::npos;
  bool isAdd = f->getName().find("__add__") != std::string::npos;
  bool isSub = f->getName().find("__sub__") != std::string::npos;
  bool isMul = f->getName().find("__mul__") != std::string::npos;
  bool isDiv = f->getName().find("__truediv__") != std::string::npos;
  bool isPow = f->getName().find("__pow__") != std::string::npos;
  if (!isEq && !isGt && !isLt && !isAdd && !isSub && !isMul && !isPow && !isDiv)
    return;

  auto *M = v->getModule();
  auto *self = M->Nr<VarValue>(pf->arg_front());
  auto *selfType = self->getType();
  auto *lhs = v->front();
  auto *rhs = v->back();
  auto *lhsType = lhs->getType();
  auto *rhsType = rhs->getType();

  bool isSqrtInv = false;
  if (isDiv) { // Special case where 1 / sqrt(x) is called
    auto *sqrtInstr = cast<CallInstr>(rhs);
    if (sqrtInstr) {
      auto *sqrtFunc = util::getFunc(sqrtInstr->getCallee());
      if (sqrtFunc)
        isSqrtInv = sqrtFunc->getName().find("sqrt") != std::string::npos;
    }
  }

  bool lhs_is_secure_container = isSharedTensor(lhsType) || isCipherTensor(lhsType);
  bool rhs_is_secure_container = isSharedTensor(rhsType) || isCipherTensor(rhsType);

  if (!lhs_is_secure_container and !rhs_is_secure_container)
    return;

  bool lhs_is_int = lhsType->is(M->getIntType());
  bool rhs_is_int = rhsType->is(M->getIntType());

  if (isMul && lhs_is_int)
    return;
  if (isMul && rhs_is_int)
    return;
  if (isDiv && lhs_is_int && !isSqrtInv)
    return;
  if (isPow && lhs_is_int)
    return;
  if (isPow && !rhs_is_int)
    return;

  std::string methodName = isEq        ? "secure_eq"
                           : isGt      ? "secure_gt"
                           : isLt      ? "secure_lt"
                           : isAdd     ? "secure_add"
                           : isSub     ? "secure_sub"
                           : isMul     ? "secure_mul"
                           : isSqrtInv ? "secure_sqrt_inv"
                           : isDiv     ? "secure_div"
                           : isPow     ? "secure_pow"
                                       : "invalid_operation";
  if (isSqrtInv) {
    rhs = cast<CallInstr>(rhs)->back();
    rhsType = rhs->getType();
  }

  auto *method = getOrRealizeSequreInternalMethod(M, methodName, {selfType, lhsType, rhsType}, {});
  if (!method) return;

  auto *func = util::call(method, {self, lhs, rhs});
  v->replaceAll(func);
}

void ExpressivenessTransformations::handle(CallInstr *v) { transform(v); }

} // namespace sequre

#include "debugger.h"
#include "helpers/utils.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"


namespace sequre {

using namespace codon::ir;

class SecureVarsSingleton {
public:
  static SecureVarsSingleton& instance() {
    static SecureVarsSingleton inst;
    return inst;
  }
  Var *get( Var *secureVar ) {
    auto *M  = secureVar->getModule();
    auto  it = map.find(secureVar->getId());
    
    if ( it != map.end() )
      return M->getVar(it->second);
    
    return nullptr;
  }
  void insert( codon::ir::id_t varId, Var *var ) {
    map.insert({varId, var->getId()});
  }
private:
  SecureVarsSingleton() = default;
  ~SecureVarsSingleton() = default;
  std::unordered_map<codon::ir::id_t, codon::ir::id_t> map;
};

void Debugger::replaceVars( Value *v, VarValue *mpc ) {
  auto *var = util::getVar(v);
  
  if ( var ) {
    if ( isSecureContainer(var->getType()) ) {
      auto *rawVar = SecureVarsSingleton::instance().get(var);

      if ( !rawVar ) {
        auto *M         = v->getModule();
        auto *reveal    = revealCall(var, mpc);
        rawVar          = M->Nr<Var>(reveal->getType(), false, false, "raw_" + var->getName());
        auto *rawAssign = M->Nr<AssignInstr>(rawVar, reveal);

        insertBefore(rawAssign);
        SecureVarsSingleton::instance().insert(var->getId(), rawVar);
      }
      
      v->replaceUsedVariable(var, rawVar);
    }
  } else {
    for ( auto *val : v->getUsedValues() )
      replaceVars(val, mpc);
  }
}

void Debugger::attachDebugger( AssignInstr *v ) {
  auto *pf = getParentFunc();
  if ( !hasDebugAttr(pf) )
    return;
  
  auto *lhs = v->getLhs();
  auto *rhs = v->getRhs();

  if ( !isSecureContainer(lhs->getType()) )
    return;
  
  auto *M      = v->getModule();
  auto *parent = cast<BodiedFunc>(pf);
  auto *body   = cast<SeriesFlow>(parent->getBody());

  auto *mpc = M->Nr<VarValue>(pf->arg_front());
  assert( isMPC(mpc) && "ERROR: The first argument of sequre function should be the MPC instance" );

  util::CloneVisitor cv(M);
  auto *rawRhs = cv.clone(rhs);
  replaceVars(rawRhs, mpc);

  auto *revealLhs = revealCall(lhs, mpc);
  auto *rawLhs    = M->Nr<Var>(revealLhs->getType(), false, false, "raw_" + lhs->getName());
  auto *rawAssign = M->Nr<AssignInstr>(rawLhs, rawRhs);
  SecureVarsSingleton::instance().insert(lhs->getId(), rawLhs);

  auto it = body->begin();
  while ( *it != v && it != body->end() ) ++it;
  body->insert(it++, rawAssign);

  auto *assertFunc = M->getOrRealizeFunc(
    "assert_eq_approx",
    {M->getStringType(), rawLhs->getType(), rawLhs->getType(), M->getFloatType(), M->getBoolType()}, {},
    "std.sequre.utils.testing");
  assert ( assertFunc && "SEQURE TYPE REALIZATION ERROR: Could not realize std.sequre.utils.testing.assert_eq_approx" );
  
  auto srcInfo     = v->getSrcInfo();
  auto srcPath     = srcInfo.file + ":" + std::to_string(srcInfo.line);
  auto *assertCall = util::call(
    assertFunc,
    {M->getString(srcPath), M->Nr<VarValue>(rawLhs), revealLhs, M->getFloat(0.02), M->getBool(false)});
  body->insert(it++, assertCall);
}

void Debugger::handle( AssignInstr *v ) { attachDebugger(v); }

} // namespace sequre

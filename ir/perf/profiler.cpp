#include "profiler.h"
// #include "codon/sir/util/cloning.h"
#include "codon/cir/util/irtools.h"
// #include "codon/sir/util/matching.h"
// #include <iterator>
// #include <math.h>

namespace sequre {

using namespace codon::ir;

const std::string perfModule = "std.perf";
const std::string perfProfileMethod = "__internal__perf_profile";

void Profiler::attachProfiler(CallInstr *v) {
  Func *f = util::getFunc(v->getCallee());
  if (!f)
    return;

  Module *M = v->getModule();
  Value *funcVal = M->Nr<VarValue>(f->getActual());
  types::Type *funcType = cast<types::FuncType>(f->getType());
  std::vector<Value *> args = {M->getString(f->getName()), funcVal};
  std::vector<types::Type *> argsTypes = {M->getStringType(), funcType};

  for (auto it = v->begin(); it != v->end(); ++it) {
    args.push_back(*it);
    argsTypes.push_back((*it)->getType());
  }
  
  auto *method = M->getOrRealizeFunc(perfProfileMethod, argsTypes, {}, perfModule);
  if (!method)
    return;

  auto *func = util::call(method, args);
  v->replaceAll(func);
}

void Profiler::handle(codon::ir::CallInstr *v) { attachProfiler(v); }

} // namespace sequre

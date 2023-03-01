#ifndef _SEQURE_UTILS
#define _SEQURE_UTILS

#include "codon/sir/util/irtools.h"

namespace sequre {

using namespace codon::ir;

const std::string sharedTensorTypeName = "SharedTensor";
const std::string cipherTensorTypeName = "CipherTensor";


bool isSequreFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

bool isPolyOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mpc_poly_opt");
}

bool isFactOptFunc(Func *f) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_mat_opt");
}

bool isCipherOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_cipher_opt");
}

bool isSharedTensor( types::Type *t ) {
  return t->getName().find(sharedTensorTypeName) != std::string::npos;
}

bool isCipherTensor( types::Type *t ) {
  return t->getName().find(cipherTensorTypeName) != std::string::npos;
}

Func *getOrRealizeSequreInternalMethod(Module *M, std::string const &methodName,
                                       std::vector<types::Type *> args,
                                       std::vector<types::Generic> generics = {}) {
  auto *sequreInternalType = M->getOrRealizeType("Internal", {}, "std.sequre.types.internal");
  return M->getOrRealizeMethod(sequreInternalType, methodName, args, generics);
}

} // namespace sequre

#endif // _SEQURE_UTILS

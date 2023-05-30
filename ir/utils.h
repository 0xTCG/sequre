#pragma once

namespace sequre {

using namespace codon::ir;

bool isSequreFunc( Func *f );
bool isPolyOptFunc( Func *f );
bool isFactOptFunc( Func *f );
bool isCipherOptFunc( Func *f );
bool isSharedTensor( types::Type *t );
bool isCipherTensor( types::Type *t );
bool isMPP( types::Type *t );

Func *getOrRealizeSequreInternalMethod( Module *M, std::string const &methodName,
                                        std::vector<types::Type *> args,
                                        std::vector<types::Generic> generics);

} // namespace sequre

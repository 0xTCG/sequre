#include "mhe.h"
#include "analysis/cipher_opt.h"
#include "helpers/utils.h"

#include <exception>
#include <iostream>
#include <vector>

namespace sequre {

using namespace codon::ir;

/* Pass */

void MHEOptimizations::run( Module *M ) {
  // The decorated functions first: realizing new operators below appends functions to the module. (@mhe_enc_opt is
  // accepted and does nothing: encrypting the public operands of products was slower than the kernel routers'
  // plaintext routes everywhere it was measured.)
  std::vector<BodiedFunc *> funcs;
  for ( auto *var : *M ) {
    auto *f = cast<BodiedFunc>(var);
    if ( f && f->getBody() && hasCipherOptAttr(f) ) funcs.push_back(f);
  }

  for ( auto *f : funcs ) {
    // (one line per realized function: a function planned twice would show twice)
    if ( irOptDebug() )
      std::cerr << "[mhe_cipher_opt] function " << f->getUnmangledName() << " (" << f->getId() << "): planned"
                << std::endl;
    try {
      applyCipherOpt(f);
    } catch ( const std::exception &e ) {
      irOptInternalError(f, e.what());
    } catch ( ... ) {
      irOptInternalError(f, "unexpected exception");
    }
  }
}

} // namespace sequre

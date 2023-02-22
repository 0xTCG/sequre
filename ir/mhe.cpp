#include "mhe.h"
#include "codon/sir/util/cloning.h"
#include "codon/sir/util/irtools.h"
#include "codon/sir/util/matching.h"
#include <iterator>
#include <math.h>

namespace sequre {

using namespace codon::ir;

const std::string cipherTensorTypeName = "CipherTensor";
const std::string builtinModule = "std.internal.builtin";

const enum Operation { add, mul, matmul, pow, noop };

/// CryptoType enumerates all possible data types in cryptographic context
/// @param cipher Ciphertext type --- encoded and ancrypted data
/// @param plain Plaintext type --- encoded data
/// @param raw Raw type --- raw data (not encrypted nor encoded)
/// @param invalid Any type that does not fit into cryptographic context
const enum CryptoType  { cipher, plain, raw, invalid };


class BETNode {
  Value      *value;
  CryptoType  cryptoType;
  Operation   operation;
  BETNode    *leftChild;
  BETNode    *rightChild;
  bool        expanded;

public:
  BETNode();
  BETNode( Value *value );
  BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild );
  BETNode( Value *value, CryptoType cryptoType, Operation operation, bool expanded );
  ~BETNode() {
    if ( leftChild )  delete leftChild;
    if ( rightChild ) delete rightChild;
  }
  BETNode *copy() const;

  Value           *getValue()       const { return value; };
  Var             *getVariable()    const { return util::getVar(value); }
  VarValue        *getVarValue()    const { return cast<VarValue>(value); };
  BETNode         *getLeftChild()   const { return leftChild; }
  BETNode         *getRightChild()  const { return rightChild; }
  CryptoType       getCryptoType()  const { return cryptoType; }
  Operation        getOperation()   const { return operation; }
  int64_t          getIntConst()    const { return util::getConst<int64_t>(value); };
  double           getDoubleConst() const { return util::getConst<double>(value); };
  codon::ir::id_t  getVariableId()  const { return getVariable()->getId(); };
  
  void setValue( Value *value )             { this->value = value; }
  void setOperation( Operation operation )  { this->operation = operation; }
  void setLeftChild( BETNode *leftChild )   { this->leftChild = leftChild; }
  void setRightChild( BETNode *rightChild ) { this->rightChild = rightChild; }
  void setExpanded()                        { expanded = true; }
  void setCryptoTypeFromValue();

  bool isLeaf()        const { return !leftChild && !rightChild; }
  bool isOperation()   const { return operation != noop; }
  bool isAdd()         const { return operation == add; }
  bool isMul()         const { return operation == mul; }
  bool isPow()         const { return operation == pow; }
  bool isCommutative() const { return isAdd() || isMul(); }
  bool isExpanded()    const { return expanded; }
  bool isCiphertext()  const { return cryptoType == cipher; }
  bool isPlaintext()   const { return cryptoType == plain; }
  
  bool checkIsVariable()            const { return bool(getVariable()); }
  bool checkIsIntConst()            const { return util::isConst<int64_t>(value); }
  bool checkIsDoubleConst()         const { return util::isConst<double>(value); }
  bool checkIsConst()               const { return checkIsIntConst() || checkIsDoubleConst(); }
  bool checkIsCipherTensor()        const;
  bool checkIsCiphertext()          const;
  bool checkIsPlaintext()           const;
  bool checkIsSameTree( BETNode * ) const;
  
  void swapChildren() { std::swap(leftChild, rightChild); }
  void replace( BETNode * );
  void realizeCryptoType();

  std::string       const getOperationIRName() const;
  void print( int ) const;
};

BETNode::BETNode()
  : value(nullptr), cryptoType(invalid), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(false) {}

BETNode::BETNode( Value *value )
  : value(value), cryptoType(invalid), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(true) {
  if ( value ) setCryptoTypeFromValue();
}

BETNode::BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild )
  : value(nullptr), cryptoType(invalid), operation(operation), leftChild(leftChild), rightChild(rightChild), expanded(false) {}

BETNode::BETNode( Value *value, CryptoType cryptoType, Operation operation, bool expanded )
  : value(value), cryptoType(cryptoType), operation(operation), leftChild(nullptr), rightChild(nullptr), expanded(expanded) {}

BETNode *BETNode::copy() const {
  auto *newNode = new BETNode(value, cryptoType, operation, expanded);
  auto *lc      = getLeftChild();
  auto *rc      = getRightChild();
  
  if ( lc ) newNode->setLeftChild(lc->copy());
  if ( rc ) newNode->setRightChild(rc->copy());
  
  return newNode;
}

void BETNode::setCryptoTypeFromValue() {
  if ( checkIsCiphertext() ) cryptoType = cipher;
  else if ( checkIsPlaintext() ) cryptoType = plain;
  else if ( checkIsVariable() || checkIsConst() ) cryptoType = raw;
  else cryptoType = invalid;
}

bool BETNode::checkIsCipherTensor() const {
  return value->getType()->getName().find(cipherTensorTypeName) != std::string::npos;
}

bool BETNode::checkIsCiphertext() const {
  if ( !checkIsCipherTensor() ) return false;
  return value->getType()->getName().find("Ciphertext") != std::string::npos;
}

bool BETNode::checkIsPlaintext() const {
  if ( !checkIsCipherTensor() ) return false;
  return value->getType()->getName().find("Plaintext") != std::string::npos;
}

bool BETNode::checkIsSameTree( BETNode *other ) const {
  if ( isLeaf() && other->isLeaf() ) {
    if ( checkIsIntConst() && other->checkIsIntConst() ) return getIntConst() == other->getIntConst();
    if ( checkIsDoubleConst() && other->checkIsDoubleConst() ) return getDoubleConst() == other->getDoubleConst();
    if ( checkIsVariable() && other->checkIsVariable() ) return getVariableId() == other->getVariableId();
  } else if ( !isLeaf() && !other->isLeaf() ) {
    if ( isOperation() && other->isOperation() && getOperation() != other->getOperation() ) return false;

    if ( getLeftChild()->checkIsSameTree(other->getLeftChild()) &&
         getRightChild()->checkIsSameTree(other->getRightChild()) ) return true;

    if ( isCommutative() )
      if ( getLeftChild()->checkIsSameTree(other->getRightChild()) &&
           getRightChild()->checkIsSameTree(other->getLeftChild()) ) return true;
  }

  return false;
}

void BETNode::replace( BETNode *other ) {
  value      = other->getValue();
  cryptoType = other->getCryptoType();
  operation  = other->getOperation();
  leftChild  = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded   = other->isExpanded();
}

void BETNode::realizeCryptoType() {
  if ( cryptoType == invalid ) {
    if ( isLeaf() ) setCryptoTypeFromValue();
    else { // Realize crypto type from children
      auto *lc = getLeftChild();
      auto *rc = getRightChild();
      
      lc->realizeCryptoType();
      rc->realizeCryptoType();

      // Not possible for lc or rc to have invalid type here
      assert( lc->getCryptoType() != invalid && "Crypto type realization error (left child type could not be realized)" );
      assert( rc->getCryptoType() != invalid && "Crypto type realization error (left child type could not be realized)" );

      if ( lc->isCiphertext() || rc->isCiphertext() ) cryptoType = cipher;
      else if ( lc->isPlaintext() || rc->isPlaintext() ) cryptoType = plain;
      else cryptoType = raw;
    }
  }

  assert( cryptoType != invalid && "Cannot realize crypto type" );
}

std::string const BETNode::getOperationIRName() const {
  if ( isAdd() ) return "__add__";
  if ( isMul() ) return "__mul__";
  if ( isPow() ) return "__pow__";
  assert(false && "BET node operator not supported in MHE IR optimizations.");
}

void BETNode::print( int level = 0 ) const {
  for (int i = 0; i < level; ++i)
    std::cout << "    ";

  std::cout << operation << " " << getVariableId()
            << ( checkIsConst() ? " Is constant " : " Not constant " )
            << value << std::endl;

  if ( leftChild ) leftChild->print(level + 1);
  if ( rightChild ) rightChild->print(level + 1);
}


class BET {
  std::unordered_map<codon::ir::id_t, BETNode *> betPerVar;

public:
  BET() {}
  ~BET() { for ( auto& it: betPerVar ) delete it.second; }

  void addBET( Var* var, BETNode *betNode ) { betPerVar[var->getId()] = betNode; }
  void expandNode( BETNode * );

  bool reduceLvl( BETNode * );
  void reduceAll( BETNode * );

  void escapePows( BETNode * );

private:
  std::pair<BETNode *, BETNode *> findFactorizationNodes( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>& );
  std::pair<BETNode *, BETNode *> findFactorsInMulTree( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode *, BETNode *, BETNode * );
  BETNode *internalIsVisited( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode * );
};

void BET::expandNode( BETNode *betNode ) {
  if ( betNode->isExpanded() ) return;

  if ( betNode->isLeaf() ) {
    auto search = betPerVar.find(betNode->getVariableId());
    if ( search != betPerVar.end() ) betNode->replace(search->second);
  } else {
    expandNode(betNode->getLeftChild());
    expandNode(betNode->getRightChild());
  }

  betNode->setExpanded();
}

bool BET::reduceLvl( BETNode *node ) {
  if ( node->isLeaf() || !node->isAdd() ) return false;

  std::vector<BETNode *> visited;
  std::unordered_map<BETNode *, std::vector<BETNode *>> metadata;
  std::pair<BETNode *, BETNode*> factors = findFactorizationNodes(node, visited, metadata);
  if ( !factors.first || !factors.second ) return false;
  
  BETNode *factor               = factors.first;
  BETNode *firstFactorParent    = metadata[factor][0];
  BETNode *firstFactorSibling   = metadata[factor][1];
  BETNode *firstFactorAncestor  = metadata[factor][2];
  BETNode *secondFactorParent   = metadata[factors.second][0];
  BETNode *secondFactorSibling  = metadata[factors.second][1];
  BETNode *secondFactorAncestor = metadata[factors.second][2];
  BETNode *vanisihingAdd        = metadata[factors.second][3];
  BETNode *vanisihingAddTail    = metadata[factors.second][4];

  // Delete secondFactorParent from secondFactorAncestor mul subtree
  secondFactorParent->replace(secondFactorSibling);
  
  // Delete firstFactorParent from firstFactorAncestor mul subtree
  firstFactorParent->replace(firstFactorSibling);

  // Replace firstFactorAncestor with the new subtree
  firstFactorAncestor->setLeftChild(factor);
  firstFactorAncestor->setRightChild(new BETNode(add, firstFactorAncestor->copy(), firstFactorAncestor));
  
  // Delete the vanishingAdd node
  vanisihingAdd->replace(vanisihingAddTail);
}

void BET::reduceAll( BETNode *root ) {
  while ( reduceLvl(root) );
}

void BET::escapePows( BETNode *node ) {
  if ( node->isLeaf() ) return;

  if ( !node->isPow() ) {
    escapePows(node->getLeftChild());
    escapePows(node->getRightChild());
    return;
  }

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();

  assert(rc->checkIsIntConst() &&
         "Sequre factorization optimization expects each exponent to be an integer constant.");
  assert(rc->getIntConst() > 0 &&
         "Sequre factorization optimization expects each exponent to be positive.");
  if ( rc->getIntConst() == 1 ) {
    node->replace(lc);
    return;
  }

  auto *newMulNode = new BETNode(mul, lc, lc);
  for (int i = 0; i < rc->getIntConst() - 2; ++i) newMulNode = new BETNode(mul, lc, newMulNode->copy());

  node->replace(newMulNode);
}

std::pair<BETNode *, BETNode *> BET::findFactorizationNodes(
    BETNode *node, std::vector<BETNode *> &visited, std::unordered_map<BETNode *, std::vector<BETNode *>> &metadata ) {
  assert(node->isAdd() && "BET: Tried to find factors in non-addition tree.");

  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();

  std::pair<BETNode *, BETNode *> factors = std::make_pair(nullptr, nullptr);
  if ( lc->isMul() ) factors = findFactorsInMulTree(lc, visited, metadata, node, lc, rc);
  else if ( lc->isAdd() ) factors = findFactorizationNodes(lc, visited, metadata);
  if ( rc->isMul()) factors = findFactorsInMulTree(rc, visited, metadata, node, rc, lc);
  else if ( rc->isAdd() ) factors = findFactorizationNodes(rc, visited, metadata);

  return factors;
}

BETNode *BET::internalIsVisited( BETNode *node, std::vector<BETNode *> &visited, std::unordered_map<BETNode *, std::vector<BETNode *>> &metadata, BETNode *firstMulAncestor ) {
  for (BETNode *n : visited) {
    if ( metadata[n][2] == firstMulAncestor ) continue;
    if ( node->checkIsSameTree(n) ) return n;
  }
  return nullptr;
}

std::pair<BETNode *, BETNode *> BET::findFactorsInMulTree(
    BETNode *node, std::vector<BETNode *> &visited,
    std::unordered_map<BETNode *, std::vector<BETNode *>> &metadata,
    BETNode *firstMulAncestor, BETNode *addAncestor, BETNode *addSibling ) {
  assert(node->isMul() && "BET: Tried to find factors in non-multiplication tree.");

  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();
  
  std::pair<BETNode *, BETNode *> factors = std::make_pair(nullptr, nullptr);

  if ( !lc->isMul() ) {
    metadata[lc].push_back(node);
    metadata[lc].push_back(rc);
    metadata[lc].push_back(firstMulAncestor);
    metadata[lc].push_back(addAncestor);
    metadata[lc].push_back(addSibling);

    if ( factors.second = internalIsVisited(lc, visited, metadata, firstMulAncestor) ) {
      factors.first = lc;
      return factors;
    } else visited.push_back(lc);
  }

  if ( !rc->isMul() ) {
    metadata[rc].push_back(node);
    metadata[rc].push_back(lc);
    metadata[rc].push_back(firstMulAncestor);
    metadata[rc].push_back(addAncestor);
    metadata[rc].push_back(addSibling);

    if ( factors.second = internalIsVisited(rc, visited, metadata, firstMulAncestor) ) {
      factors.first = rc;
      return factors;
    } else visited.push_back(rc);
  }
  
  if ( lc->isMul() ) factors = findFactorsInMulTree(lc, visited, metadata, firstMulAncestor, addAncestor, addSibling);
  if ( factors.first ) return factors;

  if ( rc->isMul() ) factors = findFactorsInMulTree(rc, visited, metadata, firstMulAncestor, addAncestor, addSibling);
  return factors;
}

bool isArithmeticOperation( Operation op ) { return op != add || op == mul || op == matmul || op == pow; }

/*
 * Auxiliary helpers
 */

bool isSequreFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.sequre");
}

Operation getOperation( CallInstr *callInstr ) {
  auto *f = util::getFunc(callInstr->getCallee());
  auto instrName = f->getName();
  if ( instrName.find("__add__") != std::string::npos ) return add;
  if ( instrName.find("__mul__") != std::string::npos ) return mul;
  if ( instrName.find("__pow__") != std::string::npos ) return pow;
  if ( instrName.find("matmul")  != std::string::npos ) return matmul;
  return noop;
}

/* BET tree manipulation */

Value *generateExpression(Module *M, BETNode *node) {
  if (node->isLeaf())
    return node->getVarValue();

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  assert(lc);
  assert(rc);

  auto *lopType = lc->getType();
  auto *ropType = rc->getType();
  auto *opFunc =
      M->getOrRealizeMethod(lopType, node->getOperationIRName(), {lopType, ropType});

  std::string const errMsg =
      node->getOperationIRName() + " not found in type " + lopType->getName();
  assert(opFunc && errMsg.c_str());

  auto *lop = generateExpression(M, lc);
  assert(lop);
  auto *rop = generateExpression(M, rc);
  assert(rop);

  auto *callIns = util::call(opFunc, {lop, rop});
  assert(callIns);
  auto *actualCallIns = callIns->getActual();
  assert(actualCallIns);

  return actualCallIns;
}

BETNode *parseArithmetic( CallInstr *callInstr ) {
  assert(callInstr->numArgs() == 2 && "Arithmetics are expected to be binary");

  auto *betNode = new BETNode();
  Operation operation = getOperation(callInstr);
  betNode->setOperation(operation);
  betNode->setIRType(callInstr->getType());
  if ( !isArithmeticOperation(operation) ) return betNode;

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);

  bool lhsIsConst = util::isConst<int64_t>(lhs) || util::isConst<double>(lhs);
  bool rhsIsConst = util::isConst<int64_t>(rhs) || util::isConst<double>(rhs);
  bool lhsIsLeaf  = ((lhs->getUsedVariables().size() == 1 && !lhsInstr) || lhsIsConst );
  bool rhsIsLeaf  = ((rhs->getUsedVariables().size() == 1 && !rhsInstr) || rhsIsConst );

  if ( lhsInstr ) betNode->setLeftChild(parseArithmetic(lhsInstr));
  else if ( lhsIsLeaf ) betNode->setLeftChild(new BETNode(lhs));
  else throw "BET: Arithmetic could not be parsed";

  if ( rhsInstr ) betNode->setRightChild(parseArithmetic(rhsInstr));
  else if ( rhsIsLeaf ) betNode->setRightChild(new BETNode(rhs));
  else throw "BET: Arithmetic could not be parsed";

  return betNode;
}

std::pair<Value *, BETNode *> minimizeCipherMult( Module *M, Value *instruction, BET *bet ) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if ( retIns ) {
    retIns->setValue(minimizeCipherMult(M, retIns->getValue(), bet).first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *assIns = cast<AssignInstr>(instruction);
  if ( assIns ) {
    auto *lhs = assIns->getLhs();
    auto *rhs = assIns->getRhs();
    auto transformedInstruction = minimizeCipherMult(M, rhs, bet);
    if ( transformedInstruction.second ) bet->addBET(lhs->getUsedVariables().front(), transformedInstruction.second);
    assIns->setRhs(transformedInstruction.first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *callInstr = cast<CallInstr>(instruction);
  if ( callInstr ) {
    if ( isArithmeticOperation(getOperation(callInstr)) ) {
      auto *betNode = parseArithmetic(callInstr);
      bet->expandNode(betNode);
      bet->reduceAll(betNode);
      // bet->reorderPriorities(betNode)
      return std::make_pair(generateExpression(M, betNode), betNode);
    } else {
      std::vector<Value *> newArgs;
      for ( auto arg = callInstr->begin(); arg < callInstr->end(); arg++ )
        newArgs.push_back(minimizeCipherMult(M, *arg, bet).first);
      callInstr->setArgs(newArgs);
      return std::make_pair(callInstr, nullptr);
    }
  }

  return std::make_pair(instruction, new BETNode(instruction->getUsedVariables().front()));
}

void transformExpressions( Module *M, SeriesFlow *series ) {
  auto *bet = new BET();
  for ( auto it = series->begin(); it != series->end(); ++it ) minimizeCipherMult(M, *it, bet);
}

/* IR passes */

void applyCipherPlainOptimizations( CallInstr *v ) {
  auto *M = v->getModule();
  auto *f = util::getFunc(v->getCallee());
  if ( !isSequreFunc(f) ) return;
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()));
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

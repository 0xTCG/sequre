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


class BETNode {
  Value       *value;
  types::Type *irType;
  Operation    operation;
  BETNode     *leftChild;
  BETNode     *rightChild;
  bool         expanded;

public:
  BETNode();
  BETNode( Value *value );
  BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild );
  BETNode( Value *value, types::Type *irType, Operation operation, bool expanded );
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
  Operation        getOperation()   const { return operation; }
  int64_t          getIntConst()    const { return util::getConst<int64_t>(value); };
  double           getDoubleConst() const { return util::getConst<double>(value); };
  codon::ir::id_t  getVariableId()  const { return getVariable()->getId(); };
  
  void setValue( Value *value )             { this->value = value; }
  void setIRType( types::Type *irType )     { this->irType = irType; }
  void setOperation( Operation operation )  { this->operation = operation; }
  void setLeftChild( BETNode *leftChild )   { this->leftChild = leftChild; }
  void setRightChild( BETNode *rightChild ) { this->rightChild = rightChild; }
  void setExpanded()                        { expanded = true; }

  bool isLeaf()        const { return !leftChild && !rightChild; }
  bool isOperation()   const { return operation != noop; }
  bool isAdd()         const { return operation == add; }
  bool isMul()         const { return operation == mul; }
  bool isPow()         const { return operation == pow; }
  bool isCommutative() const { return isAdd() || isMul(); }
  bool isExpanded()    const { return expanded; }
  
  bool checkIsVariable()            const { return bool(getVariable()); }
  bool checkIsIntConst()            const { return util::isConst<int64_t>(value); }
  bool checkIsDoubleConst()         const { return util::isConst<double>(value); }
  bool checkIsConst()               const { return checkIsIntConst() || checkIsDoubleConst(); }
  bool checkIsSameTree( BETNode * ) const;
  bool checkIsCipherTensor();
  bool checkIsCiphertext();
  bool checkIsPlaintext();
  
  void swapChildren() { std::swap(leftChild, rightChild); }
  void replace( BETNode * );
  types::Type *getOrRealizeIRType();

  std::string       const getOperationIRName() const;
  void print( int ) const;
};

BETNode::BETNode()
  : value(nullptr), irType(nullptr), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(false) {}

BETNode::BETNode( Value *value )
  : value(value), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(true) {
  if ( value ) getOrRealizeIRType();
}

BETNode::BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild )
  : value(nullptr), irType(nullptr), operation(operation), leftChild(leftChild), rightChild(rightChild), expanded(false) {}

BETNode::BETNode( Value *value, types::Type *irType, Operation operation, bool expanded )
  : value(value), irType(irType), operation(operation), leftChild(nullptr), rightChild(nullptr), expanded(expanded) {}

BETNode *BETNode::copy() const {
  auto *newNode = new BETNode(value, irType, operation, expanded);
  auto *lc      = getLeftChild();
  auto *rc      = getRightChild();
  
  if ( lc ) newNode->setLeftChild(lc->copy());
  if ( rc ) newNode->setRightChild(rc->copy());
  
  return newNode;
}

bool BETNode::checkIsCipherTensor() {
  return getOrRealizeIRType()->getName().find(cipherTensorTypeName) != std::string::npos;
}

bool BETNode::checkIsCiphertext() {
  if ( !checkIsCipherTensor() ) return false;
  return getOrRealizeIRType()->getName().find("Ciphertext") != std::string::npos;
}

bool BETNode::checkIsPlaintext() {
  if ( !checkIsCipherTensor() ) return false;
  return getOrRealizeIRType()->getName().find("Plaintext") != std::string::npos;
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
  irType     = other->getOrRealizeIRType();
  operation  = other->getOperation();
  leftChild  = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded   = other->isExpanded();
}

types::Type *BETNode::getOrRealizeIRType() {
  if ( irType ) return irType;
  if ( irType = value->getType() ) return irType;
  if ( isLeaf() ) return nullptr;

  // Realize IR type from children
  auto *lc = getLeftChild();
  auto *rc = getRightChild();

  auto *lcType = lc->getOrRealizeIRType();
  auto *rcType = rc->getOrRealizeIRType();
  
  // Not possible for lc or rc to have invalid type here
  assert( lcType && "Crypto type realization error (left child type could not be realized)" );
  assert( rcType && "Crypto type realization error (left child type could not be realized)" );

  if ( lc->checkIsCiphertext() ) irType = lcType;
  else if ( rc->checkIsCiphertext() ) irType = rcType;
  else if ( lc->checkIsPlaintext() ) irType = lcType;
  else if ( rc->checkIsPlaintext() ) irType = rcType;
  else irType = lcType;

  assert( irType && "Cannot realize crypto type" );
  return irType;
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

bool isCipherOptFunc( Func *f ) {
  return bool(f) && util::hasAttribute(f, "std.sequre.attributes.mhe_cipher_opt");
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

  auto *lopType = lc->getOrRealizeIRType();
  auto *ropType = rc->getOrRealizeIRType();
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
  
  std::cout << "DEBUG MHE PASS " << betNode->getOrRealizeIRType()->getName() << "\n";
  
  if ( !isArithmeticOperation(operation) ) return betNode;

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);

  if ( lhsInstr ) betNode->setLeftChild(parseArithmetic(lhsInstr));
  else betNode->setLeftChild(new BETNode(lhs));

  if ( rhsInstr ) betNode->setRightChild(parseArithmetic(rhsInstr));
  else betNode->setRightChild(new BETNode(rhs));

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
      // bet->reorderPriorities(betNode)  // TODO
      return std::make_pair(generateExpression(M, betNode), betNode);
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

void transformExpressions( Module *M, SeriesFlow *series ) {
  auto *bet = new BET();
  for ( auto it = series->begin(); it != series->end(); ++it ) minimizeCipherMult(M, *it, bet);
}

/* IR passes */

void applyCipherPlainOptimizations( CallInstr *v ) {
  auto *M = v->getModule();
  auto *f = util::getFunc(v->getCallee());
  if ( !isSequreFunc(f) || !isCipherOptFunc(f) ) return;
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()));
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

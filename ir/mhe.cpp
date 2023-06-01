#include "mhe.h"
#include "utils.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"
#include <iterator>
#include <math.h>

namespace sequre {

using namespace codon::ir;

const std::string builtinModule = "std.internal.builtin";

enum Operation { add, mul, matmul, pow, noop };


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
  bool isMatmul()      const { return operation == matmul; }
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

  std::string            const getOperationIRName( bool ) const;
  std::string            const getName() const;
  std::string            const getConstStr() const;
  void print( int, int ) const;
};

BETNode::BETNode()
  : value(nullptr), irType(nullptr), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(false) {}

BETNode::BETNode( Value *value )
  : value(value), operation(noop), leftChild(nullptr), rightChild(nullptr), expanded(true) {
  if ( checkIsVariable() || checkIsConst() ) getOrRealizeIRType();
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
  return isCipherTensor(getOrRealizeIRType());
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
  irType     = other->irType;
  operation  = other->getOperation();
  leftChild  = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded   = other->isExpanded();
}

types::Type *BETNode::getOrRealizeIRType() {
  // TODO: Check why `if ( irType ) return irType;` here causes irType to become silent killer of the whole code
  // Might have to do something with the this->copy() method and the way irType is copied there.

  if ( isLeaf() && (checkIsVariable() || checkIsConst()) ) {
    irType = getVariable()->getType();
    return irType;
  }
  if ( isLeaf() ) assert( irType && "Cannot realize crypto type (leaf is not a variable nor constant)" );;

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

std::string const BETNode::getOperationIRName( bool restrict = true ) const {
  if ( isAdd() ) return Module::ADD_MAGIC_NAME;
  if ( isMul() ) return Module::MUL_MAGIC_NAME;
  if ( isMatmul() ) return Module::MATMUL_MAGIC_NAME;
  if ( isPow() ) return Module::POW_MAGIC_NAME;
  if ( restrict ) assert(false && "BET node operator not supported in MHE IR optimizations.");
  else return "None";
}

std::string const BETNode::getName() const {
  if ( isOperation() ) return getOperationIRName();
  if ( checkIsVariable() ) return getVariable()->getName();
  if ( checkIsConst() )  return getConstStr();
  return "Non-parsable";
}

std::string const BETNode::getConstStr() const {
  if ( checkIsDoubleConst() ) return std::to_string(getDoubleConst());
  if ( checkIsIntConst() ) return std::to_string(getIntConst());
  return "Non-constant";
}

void BETNode::print( int level = 0, int maxLevel = 100) const {
  if ( level >= maxLevel ) return;

  for (int i = 0; i < level; ++i)
    std::cout << "    ";

  std::cout << getOperationIRName(false) << " " << ( checkIsVariable() ? getVariable()->getName() : "Non-variable" )
            << ( checkIsConst() ? " Constant " : " Non-constant " )
            << value << std::endl;

  if ( leftChild ) leftChild->print(level + 1, maxLevel);
  if ( rightChild ) rightChild->print(level + 1, maxLevel);
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
  firstFactorAncestor->setRightChild(new BETNode(add, firstFactorAncestor->copy(), secondFactorAncestor));
  firstFactorAncestor->setLeftChild(factor);
  firstFactorAncestor->setOperation(mul);
  firstFactorAncestor->setValue(nullptr);
  
  // Delete the vanishingAdd node
  vanisihingAdd->replace(vanisihingAddTail);

  return true;
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
  if ( lc->isMul() ) {
    factors = findFactorsInMulTree(lc, visited, metadata, lc, node, rc);
    if ( factors.first and factors.second ) return factors;
  }
  else if ( lc->isAdd() ) {
    factors = findFactorizationNodes(lc, visited, metadata);
    if ( factors.first and factors.second ) return factors;
  }
  if ( rc->isMul() ) {
    factors = findFactorsInMulTree(rc, visited, metadata, rc, node, lc);
    if ( factors.first and factors.second ) return factors;
  }
  else if ( rc->isAdd() ) {
    factors = findFactorizationNodes(rc, visited, metadata);
    if ( factors.first and factors.second ) return factors;
  }

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
    BETNode *firstMulAncestor, BETNode *addAncestor, BETNode *addTail ) {
  assert(node->isMul() && "BET: Tried to find factors in non-multiplication tree.");

  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();
  
  std::pair<BETNode *, BETNode *> factors = std::make_pair(nullptr, nullptr);

  if ( !lc->isMul() ) {
    metadata[lc].push_back(node);
    metadata[lc].push_back(rc);
    metadata[lc].push_back(firstMulAncestor);
    metadata[lc].push_back(addAncestor);
    metadata[lc].push_back(addTail);

    if ( (factors.second = internalIsVisited(lc, visited, metadata, firstMulAncestor)) ) {
      factors.first = lc;
      return factors;
    } else visited.push_back(lc);
  }

  if ( !rc->isMul() ) {
    metadata[rc].push_back(node);
    metadata[rc].push_back(lc);
    metadata[rc].push_back(firstMulAncestor);
    metadata[rc].push_back(addAncestor);
    metadata[rc].push_back(addTail);

    if ( (factors.second = internalIsVisited(rc, visited, metadata, firstMulAncestor)) ) {
      factors.first = rc;
      return factors;
    } else visited.push_back(rc);
  }
  
  if ( lc->isMul() ) factors = findFactorsInMulTree(lc, visited, metadata, firstMulAncestor, addAncestor, addTail);
  if ( factors.first ) return factors;

  if ( rc->isMul() ) factors = findFactorsInMulTree(rc, visited, metadata, firstMulAncestor, addAncestor, addTail);
  return factors;
}

bool isArithmeticOperation( Operation op ) { return op == add || op == mul || op == matmul || op == pow; }

/*
 * Auxiliary helpers
 */

Operation getOperation( CallInstr *callInstr ) {
  auto *f = util::getFunc(callInstr->getCallee());
  auto instrName = f->getName();
  if ( instrName.find(Module::ADD_MAGIC_NAME) != std::string::npos ) return add;
  if ( instrName.find(Module::MUL_MAGIC_NAME) != std::string::npos ) return mul;
  if ( instrName.find(Module::POW_MAGIC_NAME) != std::string::npos ) return pow;
  if ( instrName.find(Module::MATMUL_MAGIC_NAME)  != std::string::npos ) return matmul;
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

  auto *opFunc = M->getOrRealizeMethod(
    lopType, node->getOperationIRName(), {lopType, ropType});

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
  if ( callInstr->numArgs() != 2 ) {
    std::cout << "SEQURE PARSER ERROR: Arithmetic operation is not binary."
              << "\n\t\tOperation: " << util::getFunc(callInstr->getCallee())->getName()
              << "\n\t\tArgs:";
    
    for ( auto it = callInstr->begin(); it != callInstr->end(); ++it )
      std::cout << "\n\t\t\t" << (*it)->getType()->getName();
    
    std::cout << std::endl;
    assert(false && "Arithmetics are expected to be binary");
  }

  auto *betNode = new BETNode(callInstr);
  Operation operation = getOperation(callInstr);
  betNode->setOperation(operation);
  
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
  if ( !isCipherOptFunc(f) ) return;
  transformExpressions(M, cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()));
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

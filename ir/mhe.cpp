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
const enum Operation { add, mul, matmul, pow, terminal, invalid };
const enum NodeType { cipher, plain, constant, op, invalid };

/*
 * Binary expression tree
 */
class BETNode {
  int64_t    value;
  Var       *variable;
  NodeType   nodeType;
  Operation  operation;
  BETNode   *leftChild;
  BETNode   *rightChild;
  bool       expanded;

public:
  BETNode();
  BETNode( Var *variable );
  BETNode( int64_t value );
  BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild );
  ~BETNode() {
    if ( leftChild )  delete leftChild;
    if ( rightChild ) delete rightChild;
  }
  BETNode *copy() const;

  int64_t      getValue() const { return value; }
  Var         *getVariable() const { return variable; }
  VarValue    *getVarValue( Module * ) const;
  int          getVariableId() const;
  types::Type *getVariableType();
  Operation    getOperation() const { return operation; }
  NodeType     getNodeType() const { return nodeType; }
  BETNode     *getLeftChild() const { return leftChild; }
  BETNode     *getRightChild() const { return rightChild; }
  std::string const getOperationIRName() const;
  
  void setVariable( Var *variable ) { this->variable = variable; }
  void setOperation( Operation operation ) { this->operation = operation; }
  void setLeftChild( BETNode *leftChild ) { this->leftChild = leftChild; }
  void setRightChild( BETNode *rightChild ) { this->rightChild = rightChild; }
  void setExpanded() { expanded = true; }
  void setNodeTypeFromVarType();

  bool isLeaf() const { return !leftChild && !rightChild; }
  bool isOperation() const { return operation != invalid; }
  bool isAdd() const { return operation == add; }
  bool isMul() const { return operation == mul; }
  bool isPow() const { return operation == pow; }
  bool isConstant() const { return nodeType == constant; }
  bool isVariable() const { return bool(variable); }
  bool isExpanded() conts { return expanded; }
  bool isSameSubTree( BETNode * ) const;

  void replace( BETNode * );
  void swapChildren() { std::swap(leftChild, rightChild); }
  
  void print( int ) const;
};

BETNode::BETNode()
  : value(0), variable(nullptr), nodeType(invalid), operation(invalid), leftChild(nullptr), rightChild(nullptr), expanded(false) {}

BETNode::BETNode( int64_t value )
  : value(value), variable(nullptr), nodeType(constant), operation(invalid), leftChild(nullptr), rightChild(nullptr), expanded(true) {}

BETNode::BETNode( Var *variable )
  : value(0), variable(variable), nodeType(invalid), operation(invalid), leftChild(nullptr), rightChild(nullptr), expanded(true) {
  if ( variable ) setNodeTypeFromVarType();
}

BETNode::BETNode( Operation operation, BETNode *leftChild, BETNode *rightChild ) {
  : value(0), variable(nullptr), nodeType(op), operation(operation), leftChild(leftChild), rightChild(rightChild), expanded(false) {}

BETNode::BETNode( int64_t value, Var *variable, NodeType nodeType, Operation operation, bool expanded )
  : value(value), variable(variable), nodeType(nodeType), operation(operation), leftChild(nullptr), rightChild(nullptr), expanded(expanded) {}

VarValue *BETNode::getVarValue( Module *M ) const {
  Var *var = getVariable();
  assert(var);
  
  VarValue *arg = M->Nr<VarValue>(var);
  assert(arg);
  
  return arg;
}

int BETNode::getVariableId() const {
  assert(isVariable() && "GetVariableId called on non-variable node.");
  return variable->getId();
}

bool BETNode::isSameSubTree( BETNode *other ) const {
  if ( isLeaf() && other->isLeaf() ) {
    if ( isConstant() && other->isConstant() ) return getValue() == other->getValue();
    if ( isVariable() && other->isVariable() ) return getVariableId() == other->getVariableId();
  } else if ( !isLeaf() && !other->isLeaf() ) {
    if ( isOperation() && other->isOperation() && getOperation() != other->getOperation() ) return false;

    if ( getLeftChild()->isSameSubTree(other->getLeftChild()) &&
         getRightChild()->isSameSubTree(other->getRightChild()) ) return true;

    if ( isMul() || isAdd() )
      if ( getLeftChild()->isSameSubTree(other->getRightChild()) &&
           getRightChild()->isSameSubTree(other->getLeftChild()) ) return true;
  }

  return false;
}

void BETNode::replace( BETNode *other ) {
  value      = other->getValue();
  variable   = other->getVariable();
  nodeType   = other->getNodeType();
  operation  = other->getOperation();
  leftChild  = other->getLeftChild();
  rightChild = other->getRightChild();
}

BETNode *BETNode::copy() const {
  auto *newNode = new BETNode(value, variable, operation, nodeType, expanded);
  auto *lc      = getLeftChild();
  auto *rc      = getRightChild();
  if ( lc ) newNode->setLeftChild(lc->copy());
  if ( rc ) newNode->setRightChild(rc->copy());
  return newNode;
}

void BETNode::print( int level = 0 ) const {
  for (int i = 0; i < level; ++i)
    std::cout << "    ";

  std::cout << op << " " << getVariableId()
            << ( isConstant() ? " Is constant " : " Not constant " )
            << value << std::endl;

  if ( leftChild ) leftChild->print(level + 1);
  if ( rightChild ) rightChild->print(level + 1);
}

types::Type *BETNode::getType() {
  if ( !nodeType ) {
    if ( isConstant() ) nodeType = constant;
    else if ( isLeaf() ) setNodeTypeFromVarType();
    else setNodeTypeFromChildren();
  }

  assert(nodeType);
  return nodeType;
};

std::string const BETNode::getOperatorIRName() const {
  if ( isAdd() ) return "__add__";
  if ( isMul() ) return "__mul__";
  if ( isPow() ) return "__pow__";
  assert(false && "BET node operator not supported in MHE IR optimizations.");
};

bool BETNode::isCiphertextVar() {
  return variable->getType()->getName().find("Ciphertext") != std::string::npos
}

bool BETNode::isPlaintextVar() {
  return variable->getType()->getName().find("Plaintext") != std::string::npos
}

bool BETNode::isAtomicVar() {
  return variable->getType()->isAtomic()
}

void BETNode::setNodeTypeFromChildren() {
  if ( isCiphertextVar() ) nodeType = cipher;
  else if ( isPlaintextVar() ) nodeType = plain;
  else if ( isAtomicVar() ) nodeType = constant;
  else nodeType = invalid;
}

class BET {
  std::unordered_map<Var *, BETNode *> betPerVar;

public:
  BET() {}
  ~BET() { for ( auto& it: betPerVar ) delete it.second; }

  void addBET( Var* var, BETNode *betNode ) { betPerVar[var] = betNode; }

  bool reduceLvl( BETNode * );
  void reduceAll( BETNode * );

  void escapePows( BETNode * );

private:
  void addVar( int varId ) { vars.insert(varId); }
  void expandNode( BETNode * );
  void collapseDistributive( BETNode *, bool, bool, bool, bool, int, int );
  void collapseMul( BETNode *, bool, bool, bool, bool );
  void collapseMatmul( BETNode *, bool, bool, bool, bool );
};

void BET::expandNode( BETNode *betNode ) {
  if ( betNode->isExpanded() ) return;

  if ( betNode->isLeaf() ) {
    auto search = roots.find(betNode->getVariable());
    if ( search != roots.end() ) betNode->replace(search->second);
  } else {
    expandNode(betNode->getLeftChild());
    expandNode(betNode->getRightChild());
  }

  betNode->setExpanded();
}

void BET::collapseDistributive(
    BETNode *betNode, bool llc_lrc, bool llc_rrc,
    bool rlc_lrc, bool rlc_rrc, int weakOp, int strongOp ) {
  assert(weakOp < strongOp);

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  auto *llc = lc->getLeftChild();
  auto *rlc = lc->getRightChild();
  auto *lrc = rc->getLeftChild();
  auto *rrc = rc->getRightChild();

  BETNode *collapseNode;
  BETNode *leftOtherNode, *rightOtherNode;

  if ( llc_lrc ) {
    collapseNode   = llc;
    leftOtherNode  = rlc;
    rightOtherNode = rrc;
  } else if ( rlc_rrc ) {
    collapseNode   = rlc;
    leftOtherNode  = llc;
    rightOtherNode = lrc;
  } else if ( llc_rrc ) {
    collapseNode   = llc;
    leftOtherNode  = rlc;
    rightOtherNode = lrc;
  } else if ( rlc_lrc ) {
    collapseNode   = rlc;
    leftOtherNode  = llc;
    rightOtherNode = rrc;
  } else assert(false && "Non-reducible expression cannot be collapsed.");

  BETNode *surviveNode = llc_lrc ? rc : lc;
  BETNode *replaceNode = llc_lrc ? lc : rc;

  betNode->setOperator(strongOp);
  surviveNode->setOperator(weakOp);
  surviveNode->setLeftChild(leftOtherNode);
  surviveNode->setRightChild(rightOtherNode);

  replaceNode->replace(collapseNode);
}

void BET::collapseMul(
    BETNode *betNode, bool llc_lrc, bool llc_rrc, bool rlc_lrc, bool rlc_rrc) {
  collapseDistributive(betNode, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc, BET_ADD_OP, BET_MUL_OP);
}

void BET::collapseMatmul(
    BETNode *betNode, bool llc_lrc, bool llc_rrc, bool rlc_lrc, bool rlc_rrc) {
  assert(llc_lrc || rlc_rrc);
  collapseDistributive(betNode, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc, BET_ADD_OP, BET_MATMUL_OP);
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

  assert(rc->isConstant() &&
         "Sequre factorization optimization expects each exponent to be a constant.");
  assert(rc->getValue() > 0 &&
         "Sequre factorization optimization expects each exponent to be positive.");

  auto *newMulNode = new BETNode(mul, lc, lc);

  if ( rc->getValue() == 1 ) newMulNode->setRightChild(new BETNode(1));
  for (int i = 0; i < rc->getValue() - 2; ++i) newMulNode = new BETNode(mul, lc, newMulNode->copy());

  node->replace(newMulNode);
}

bool BET::internalIsVisited( BETNode *node, std::vector<BETNode *> &visited ) {
  for (BETNode *n : visited)
    if ( node->isSameSubTree(n) ) return true;
  return false;
}

BETNode *BET::findFactorInMulTree( BETNode *node, std::vector<BETNode *> &visited ) {
  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();
  
  assert(node->isMul(), "BET: Tried to find factors in non-multiplication tree.")
  BETNode *factor = nullptr;

  if ( lc->isMul() ) factor = findFactorInMulTree(lc, visited);
  else if internalIsVisited(lc, visited) return lc;
  else visited.push_back(lc);
  
  if ( rc->isMul() ) factor = findFactorInMulTree(rc, visited);
  else if internalIsVisited(rc, visited) return rc;
  else visited.push_back(rc);

  return factor;
}

BETNode *BET::findFactor( BETNode *node, std::vector<BETNode *> &visited ) {
  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();
  
  assert(node->isAdd(), "BET: Tried to find factors in non-addition tree.")

  BETNode *factor = nullptr;
  if ( lc->isMul() ) factor = findFactorInMulTree(lc, visited);
  else if ( lc->isAdd() ) factor = findFactor(lc, visited);
  if ( rc->isMul()) factor = findFactorInMulTree(rc, visited);
  else if ( rc->isAdd() ) factor = findFactor(rs, visited);

  return factor;
  }
}

void removeFactor(BETNode *node, BETNode *factor) {
  BETNode *lc = node->getLeftChild();
  BETNode *rc = node->getRightChild();

  if ( node->isLeaf() ) return;
  
  if ( node->isMul() ) {
    if ( lc->isSameSubTree(factor) ) {
      node->replace(rc);
      return;
    }
    else if ( rc->isSameSubTree(factor) ) {
      node->replace(lc);
      return;
    } else if ( lc->isMul() ) removeFactor(lc);
    else if ( rc->isMul() ) removeFactor(rc);
  }  
  
  if ( node->isAdd() ) {
    removeFactor(lc);
    removeFactor(rc);
  }
}

bool BET::reduceLvl( BETNode *node ) {
  if ( node->isLeaf() || !node->isAdd() ) return false;

  BETNode *factor = findFactor(node);
  if ( !factor ) return false;
  removeFactor(node, factor);

  setLeftChild(node);
  setRightChild(*factors.begin());
  setOperation(mul);
}

void BET::reduceAll( BETNode *root ) {
  while ( reduceLvl(root) );
}

bool isArithmeticOperation( Operation op ) { return op == add || op == mul || op == matmul || op == pow; }
bool isTerminalOperation( Operation op ) { return op == terminal; }

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
  if ( instrName.find("matmul") != std::string::npos ) return matmul;
  if ( instrName.find("reveal") != std::string::npos ) return terminal;
  return invalid;
}

/* BET tree manipulation */

Value *generateExpression(Module *M, BETNode *node) {
  if (node->isLeaf())
    return node->getVarValue(M);

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  assert(lc);
  assert(rc);

  auto *lopType = lc->getType();
  auto *ropType = rc->getType();
  auto *opFunc =
      M->getOrRealizeMethod(lopType, node->getOperatorIRName(), {lopType, ropType});

  std::string const errMsg =
      node->getOperatorIRName() + " not found in type " + lopType->getName();
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
  // Arithmetics are binary
  auto *betNode = new BETNode();
  betNode->setOperation(getOperation(callInstr));

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);

  if ( isAtomic() ) betNode->setLeftChild(new BETNode(lhs->getVal()));
  else if ( lhsInstr ) { }
  else if ( isVar ) betNode->setLeftChild(new BETNode(lhs->getUsedVariables().front()));
  else betNode->setLeftChild(parseArithmetic(lhsInstr));

  if ( rhs->isAtomic() ) betNode->setRightChild(new BETNode(rhs->getVal()));
  else if ( lhsInstr ) { }
  else if ( isVar ) betNode->setRightChild(new BETNode(rhs->getUsedVariables().front()));
  else betNode->setRightChild(parseArithmetic(rhsInstr));

  return betNode;
}

std::pair<codon::ir::Value *, codon::ir::Value *> transformInstruction( codon::ir::Value *instruction, BET *bet ) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if ( retIns ) {
    retIns->setValue(transformInstruction(retIns->getValue(), bet).first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *assIns = cast<AssignInstr>(instruction);
  if ( assIns ) {
    auto *lhs = assIns->getLhs();
    auto *rhs = assIns->getRhs();
    auto transformedInstruction = transformInstruction(rhs, bet);
    if ( transformedInstruction.second ) bet->appendBET(lhs, betNode);
    assIns->setRhs(transformedInstruction.first);
    return std::make_pair(nullptr, nullptr);
  }

  auto *callInstr = cast<CallInstr>(instruction);
  if ( callInstr ) {
    auto op = getOperation(callInstr);
    if ( isArithmetic(op) ) {
      auto *betNode = parseArithmetic(callInstr);
      betNode->expandNode();
      betNode->collapseAll();
      betNode->reorderPriorities();
      return std::make_pair(generateExpression(betNode), betNode);
    } else {
      std::vector<Value *> newArgs;
      for ( auto arg = callInstr->begin(); arg < callInstr->end(); arg++ )
        newArgs.push_back(transformInstruction(*arg, bet).first);
      callInstr->setArgs(newArgs);
      return std::make_pair(callInstr, nullptr);
    }
  }

  return std::make_pair(nullptr, nullptr);
}

void transformExpressions( SeriesFlow *series ) {
  auto *bet = new BET();
  for ( auto it = series->begin(); it != series->end(); ++it ) transformInstruction(*it, bet);
}

/* IR passes */

void applyCipherPlainOptimizations( CallInstr *v ) {
  auto *f = util::getFunc(v->getCallee());
  if ( !isSequreFunc(f) ) return;
  transformExpressions(cast<SeriesFlow>(cast<BodiedFunc>(f)->getBody()););
}

void MHEOptimizations::handle( CallInstr *v ) { applyCipherPlainOptimizations(v); }

} // namespace sequre

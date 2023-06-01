#include "mpc.h"
#include "utils.h"
#include "codon/cir/util/cloning.h"
#include "codon/cir/util/irtools.h"
#include "codon/cir/util/matching.h"
#include <iterator>
#include <math.h>

namespace sequre {

using namespace codon::ir;

const std::string builtinModule = "std.internal.builtin";

/*
 * Binary expression tree
 */
const int BET_ADD_OP = 1;
const int BET_MUL_OP = 2;
const int BET_POW_OP = 3;
const int BET_MATMUL_OP = 4;
const int BET_REVEAL_OP = 5;
const int BET_OTHER_OP = 6;
const int BET_ARITHMETICS_OP_THRESHOLD = BET_REVEAL_OP;

class MPCBETNode {
  int64_t value;
  Var *variable;
  types::Type *nodeType;
  int op;
  MPCBETNode *leftChild;
  MPCBETNode *rightChild;
  bool expanded;
  bool constant;

public:
  MPCBETNode();
  MPCBETNode(Var *variable);
  MPCBETNode(Var *variable, int op, bool expanded, int64_t value, bool constant);
  MPCBETNode(Var *variable, int op, MPCBETNode *leftChild, MPCBETNode *rightChild, bool expanded,
          int64_t value, bool constant);
  ~MPCBETNode() {
    if (leftChild)
      delete leftChild;
    if (rightChild)
      delete rightChild;
  }

  void setVariable(Var *variable) { this->variable = variable; }
  void setOperator(int op) { this->op = op; }
  void setLeftChild(MPCBETNode *leftChild) { this->leftChild = leftChild; }
  void setRightChild(MPCBETNode *rightChild) { this->rightChild = rightChild; }
  void swapChildren() { std::swap(leftChild, rightChild); }
  void setExpanded() { expanded = true; }
  Var *getVariable() const { return variable; }
  VarValue *getVarValue(Module *) const;
  int getVariableId() const { return variable ? variable->getId() : 0; }
  int getOperator() const { return op; }
  int64_t getValue() const { return value; }
  MPCBETNode *getLeftChild() const { return leftChild; }
  MPCBETNode *getRightChild() const { return rightChild; }
  bool isExpanded() const { return expanded; }
  bool isLeaf() const { return !leftChild && !rightChild; }
  bool isAdd() const { return op == BET_ADD_OP; }
  bool isMul() const { return op == BET_MUL_OP; }
  bool isPow() const { return op == BET_POW_OP; }
  bool isConstant() const { return constant; }
  bool isSameSubTree(MPCBETNode *) const;
  bool isElemWiseOp() const { return isAdd() || isMul(); }
  bool isMatmul() const { return op == BET_MATMUL_OP; }
  bool hasAtomicType() { return getType()->isAtomic(); }
  void replace(MPCBETNode *);
  MPCBETNode *copy() const;
  void print(int) const;

  types::Type *getType();
  std::string const getOperatorIRName() const;
};

MPCBETNode::MPCBETNode()
    : value(1), variable(nullptr), nodeType(nullptr), op(0), leftChild(nullptr),
      rightChild(nullptr), expanded(false), constant(false) {}

MPCBETNode::MPCBETNode(Var *variable)
    : value(1), variable(variable), op(0), leftChild(nullptr), rightChild(nullptr),
      expanded(false), constant(false) {
  if (variable)
    nodeType = variable->getType();
  else
    nodeType = nullptr;
}

MPCBETNode::MPCBETNode(Var *variable, int op, bool expanded, int64_t value, bool constant)
    : value(value), variable(variable), op(op), leftChild(nullptr), rightChild(nullptr),
      expanded(expanded), constant(constant) {
  if (variable)
    nodeType = variable->getType();
  else
    nodeType = nullptr;
}

MPCBETNode::MPCBETNode(Var *variable, int op, MPCBETNode *leftChild, MPCBETNode *rightChild,
                 bool expanded, int64_t value, bool constant)
    : value(value), variable(variable), op(op), leftChild(leftChild),
      rightChild(rightChild), expanded(expanded), constant(constant) {
  if (variable)
    nodeType = variable->getType();
  else
    nodeType = nullptr;
}

VarValue *MPCBETNode::getVarValue(Module *M) const {
  auto *var = getVariable();
  assert(var);
  auto *arg = M->Nr<VarValue>(var);
  assert(arg);
  return arg;
}

bool MPCBETNode::isSameSubTree(MPCBETNode *other) const {
  if (isLeaf() && other->isLeaf()) {
    if (isConstant() && other->isConstant())
      return getValue() == other->getValue();

    assert(variable &&
           "MPCBET leaf is neither constant nor variable. (This is internal bug within IR "
           "optimizations. Please report it to code owners.)");

    int varId = getVariableId();
    int otherVarId = other->getVariableId();
    if (varId && otherVarId)
      return varId == otherVarId;
  } else if (!isLeaf() && !other->isLeaf()) {
    if (getOperator() != other->getOperator())
      return false;

    if (getLeftChild()->isSameSubTree(other->getLeftChild()) &&
        getRightChild()->isSameSubTree(other->getRightChild()))
      return true;

    if (isMul() || isAdd())
      if (getLeftChild()->isSameSubTree(other->getRightChild()) &&
          getRightChild()->isSameSubTree(other->getLeftChild()))
        return true;
  }

  return false;
}

void MPCBETNode::replace(MPCBETNode *other) {
  variable = other->getVariable();
  op = other->getOperator();
  leftChild = other->getLeftChild();
  rightChild = other->getRightChild();
  expanded = other->isExpanded();
  value = other->getValue();
  constant = other->isConstant();
}

MPCBETNode *MPCBETNode::copy() const {
  auto *newNode = new MPCBETNode(variable, op, expanded, value, constant);
  auto *lc = getLeftChild();
  auto *rc = getRightChild();
  if (lc)
    newNode->setLeftChild(lc->copy());
  if (rc)
    newNode->setRightChild(rc->copy());
  return newNode;
}

void MPCBETNode::print(int level = 0) const {
  for (int i = 0; i < level; ++i)
    std::cout << "    ";

  std::cout << op << " " << getVariableId()
            << (constant ? " Is constant " : " Not constant ") << value << std::endl;

  if (leftChild)
    leftChild->print(level + 1);
  if (rightChild)
    rightChild->print(level + 1);
}

types::Type *MPCBETNode::getType() {
  if (!nodeType) {
    if (isConstant())
      nodeType = new types::IntType();
    else if (isLeaf())
      nodeType = variable->getType();
    else
      nodeType = getLeftChild()->getType();
  }

  assert(nodeType);
  return nodeType;
};

std::string const MPCBETNode::getOperatorIRName() const {
  if (isAdd())
    return "__add__";
  if (isMul())
    return "__mul__";
  if (isPow())
    return "__pow__";
  assert(false && "MPCBET node operator not supported in IR optimizations.");
};

class MPCBET {
  std::unordered_map<int, MPCBETNode *> roots;
  std::vector<int> stopVarIds;
  std::set<int> vars;
  std::vector<MPCBETNode *> polynomials;
  std::vector<std::vector<int64_t>> pascalMatrix;
  bool treeAltered;

public:
  MPCBET() : treeAltered(false) {}
  ~MPCBET() {
    auto *root = this->root();
    if (root)
      delete root;
  }

  int getVarsSize() const { return vars.size(); }
  void addRoot(MPCBETNode *betNode) { roots[betNode->getVariableId()] = betNode; }
  void addRoot(Var *, int);
  void addNode(MPCBETNode *);
  void addStopVar(int varId) { stopVarIds.insert(stopVarIds.begin(), varId); }
  void formPolynomials();
  void parseVars(MPCBETNode *);
  MPCBETNode *root();
  MPCBETNode *polyRoot() const;
  MPCBETNode *getNextPolyNode();
  std::vector<MPCBETNode *> generateFactorizationTrees(int);
  std::vector<int64_t> extractCoefficents(MPCBETNode *) const;
  std::vector<int64_t> extractExponents(MPCBETNode *) const;
  std::set<int> extractVars(MPCBETNode *) const;
  std::vector<std::vector<int64_t>> getPascalMatrix() const { return pascalMatrix; }

  bool expandLvl(MPCBETNode *);
  bool reduceLvl(MPCBETNode *);
  void expandAll(MPCBETNode *);
  void reduceAll(MPCBETNode *);

  void escapePows(MPCBETNode *);

private:
  void addVar(int varId) { vars.insert(varId); }
  void expandNode(MPCBETNode *);
  void expandPow(MPCBETNode *);
  void expandMul(MPCBETNode *);
  void expandMatmul(MPCBETNode *);
  void expandDistributive(MPCBETNode *, int, int);
  void collapseDistributive(MPCBETNode *, bool, bool, bool, bool, int, int);
  void collapseMul(MPCBETNode *, bool, bool, bool, bool);
  void collapseMatmul(MPCBETNode *, bool, bool, bool, bool);
  void formPolynomial(MPCBETNode *);
  void extractCoefficents(MPCBETNode *, std::vector<int64_t> &) const;
  void extractExponents(MPCBETNode *, std::vector<int64_t> &) const;
  void extractVars(MPCBETNode *, std::set<int> &) const;
  void parseExponents(MPCBETNode *, std::map<int, int64_t> &) const;
  void updatePascalMatrix(int64_t);
  MPCBETNode *getMulTree(MPCBETNode *, MPCBETNode *, int64_t, int64_t);
  MPCBETNode *getPowTree(MPCBETNode *, MPCBETNode *, int64_t, int64_t);
  int64_t parseCoefficient(MPCBETNode *) const;
  int64_t getBinomialCoefficient(int64_t, int64_t);
  std::vector<int64_t> getPascalRow(int64_t);
};

void MPCBET::expandNode(MPCBETNode *betNode) {
  if (betNode->isExpanded())
    return;

  int op = betNode->getOperator();
  if (!op) {
    auto search = roots.find(betNode->getVariableId());
    if (search != roots.end())
      betNode->replace(search->second);
  } else {
    expandNode(betNode->getLeftChild());
    expandNode(betNode->getRightChild());
  }

  betNode->setExpanded();
}

void MPCBET::expandPow(MPCBETNode *betNode) {
  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  assert(rc->isConstant() &&
         "Sequre polynomial optimization expects each exponent to be a constant.");

  if (lc->isMul()) {
    treeAltered = true;
    betNode->setOperator(BET_MUL_OP);
    lc->setOperator(BET_POW_OP);
    auto *newPowNode =
        new MPCBETNode(nullptr, BET_POW_OP, lc->getRightChild(), rc, true, 1, false);
    betNode->setRightChild(newPowNode);
    lc->setRightChild(rc->copy());
    return;
  }

  if (lc->isAdd()) {
    treeAltered = true;
    auto *v1 = lc->getLeftChild();
    auto *v2 = lc->getRightChild();

    auto *powTree = getPowTree(v1, v2, rc->getValue(), 0);

    betNode->setOperator(BET_ADD_OP);
    delete lc;
    betNode->setLeftChild(powTree->getLeftChild());
    delete rc;
    betNode->setRightChild(powTree->getRightChild());
    return;
  }
}

void MPCBET::expandDistributive(MPCBETNode *betNode, int weakOp, int strongOp) {
  assert(weakOp < strongOp);
  treeAltered = true;

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  auto *addNode = lc->isAdd() ? lc : rc;
  auto *otherNode = lc->isAdd() ? rc : lc;
  betNode->setOperator(weakOp);
  addNode->setOperator(strongOp);
  auto *newMulNode = new MPCBETNode(nullptr, strongOp, addNode->getRightChild(),
                                 otherNode, true, 1, false);
  addNode->setRightChild(otherNode->copy());
  
  if (lc == otherNode) {
    newMulNode->swapChildren();
    addNode->swapChildren();
    betNode->setLeftChild(newMulNode);
  }
  if (rc == otherNode)
    betNode->setRightChild(newMulNode);
}

void MPCBET::expandMul(MPCBETNode *betNode) {
  expandDistributive(betNode, BET_ADD_OP, BET_MUL_OP);
}

void MPCBET::expandMatmul(MPCBETNode *betNode) {
  expandDistributive(betNode, BET_ADD_OP, BET_MATMUL_OP);
}

void MPCBET::collapseDistributive(MPCBETNode *betNode, bool llc_lrc, bool llc_rrc, bool rlc_lrc,
                      bool rlc_rrc, int weakOp, int strongOp) {
  assert(weakOp < strongOp);

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  auto *llc = lc->getLeftChild();
  auto *rlc = lc->getRightChild();
  auto *lrc = rc->getLeftChild();
  auto *rrc = rc->getRightChild();

  MPCBETNode *collapseNode;
  MPCBETNode *leftOtherNode, *rightOtherNode;

  if (llc_lrc) {
    collapseNode = llc;
    leftOtherNode = rlc;
    rightOtherNode = rrc;
  } else if (rlc_rrc) {
    collapseNode = rlc;
    leftOtherNode = llc;
    rightOtherNode = lrc;
  } else if (llc_rrc) {
    collapseNode = llc;
    leftOtherNode = rlc;
    rightOtherNode = lrc;
  } else if (rlc_lrc) {
    collapseNode = rlc;
    leftOtherNode = llc;
    rightOtherNode = rrc;
  } else assert(false && "Non-reducible expression cannot be collapsed.");

  treeAltered = true;

  MPCBETNode *surviveNode = llc_lrc ? rc : lc;
  MPCBETNode *replaceNode = llc_lrc ? lc : rc;

  betNode->setOperator(strongOp);
  surviveNode->setOperator(weakOp);
  surviveNode->setLeftChild(leftOtherNode);
  surviveNode->setRightChild(rightOtherNode);

  replaceNode->replace(collapseNode);
}

void MPCBET::collapseMul(MPCBETNode *betNode, bool llc_lrc, bool llc_rrc, bool rlc_lrc,
                      bool rlc_rrc) {
  collapseDistributive(betNode, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc, BET_ADD_OP, BET_MUL_OP);
}

void MPCBET::collapseMatmul(MPCBETNode *betNode, bool llc_lrc, bool llc_rrc, bool rlc_lrc,
                      bool rlc_rrc) {
  assert(llc_lrc || rlc_rrc);
  collapseDistributive(betNode, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc, BET_ADD_OP, BET_MATMUL_OP);
}

void MPCBET::formPolynomial(MPCBETNode *betNode) {
  if (betNode->isLeaf())
    return;

  if (betNode->isPow()) {
    expandPow(betNode);
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  if (!betNode->isMul() || !(lc->isAdd() || rc->isAdd())) {
    formPolynomial(lc);
    formPolynomial(rc);
    return;
  }

  expandMul(betNode);
}

void MPCBET::extractCoefficents(MPCBETNode *betNode,
                             std::vector<int64_t> &coefficients) const {
  if (!(betNode->isAdd())) {
    coefficients.push_back(parseCoefficient(betNode));
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractCoefficents(lc, coefficients);
  extractCoefficents(rc, coefficients);
}

void MPCBET::extractExponents(MPCBETNode *betNode, std::vector<int64_t> &exponents) const {
  if (!(betNode->isAdd())) {
    std::map<int, int64_t> termExponents;
    for (auto varId : vars)
      termExponents[varId] = 0;
    parseExponents(betNode, termExponents);
    for (auto e : termExponents)
      exponents.push_back(e.second);
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractExponents(lc, exponents);
  extractExponents(rc, exponents);
}

void MPCBET::extractVars(MPCBETNode *betNode, std::set<int> &vars) const {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    vars.insert(betNode->getVariableId());
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();
  extractVars(lc, vars);
  extractVars(rc, vars);
}

void MPCBET::parseExponents(MPCBETNode *betNode,
                         std::map<int, int64_t> &termExponents) const {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    termExponents[betNode->getVariableId()]++;
    return;
  }

  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  if (betNode->isPow() && !lc->isConstant()) {
    assert(rc->isConstant() &&
           "Sequre polynomial optimization expects each exponent to be a constant.");
    termExponents[lc->getVariableId()] += rc->getValue();
    return;
  }

  parseExponents(lc, termExponents);
  parseExponents(rc, termExponents);
}

void MPCBET::updatePascalMatrix(int64_t n) {
  for (auto i = pascalMatrix.size(); i < n + 1; ++i) {
    auto newRow = std::vector<int64_t>(i + 1);
    for (auto j = 0; j < i + 1; ++j)
      newRow[j] = (j == 0 || j == i)
                      ? 1
                      : (pascalMatrix[i - 1][j - 1] + pascalMatrix[i - 1][j]);
    pascalMatrix.push_back(newRow);
  }
}

MPCBETNode *MPCBET::getMulTree(MPCBETNode *v1, MPCBETNode *v2, int64_t constant, int64_t iter) {
  auto *pascalNode =
      new MPCBETNode(nullptr, 0, true, getBinomialCoefficient(constant, iter), true);
  auto *leftConstNode = new MPCBETNode(nullptr, 0, true, constant - iter, true);
  auto *rightConstNode = new MPCBETNode(nullptr, 0, true, iter, true);
  auto *leftPowNode =
      new MPCBETNode(nullptr, BET_POW_OP, v1->copy(), leftConstNode, true, 1, false);
  auto *rightPowNode =
      new MPCBETNode(nullptr, BET_POW_OP, v2->copy(), rightConstNode, true, 1, false);
  auto *rightMulNode =
      new MPCBETNode(nullptr, BET_MUL_OP, leftPowNode, rightPowNode, true, 1, false);

  return new MPCBETNode(nullptr, BET_MUL_OP, pascalNode, rightMulNode, true, 1, false);
}

MPCBETNode *MPCBET::getPowTree(MPCBETNode *v1, MPCBETNode *v2, int64_t constant, int64_t iter) {
  auto *newMulNode = getMulTree(v1, v2, constant, iter);

  if (constant == iter)
    return newMulNode;

  auto *newAddNode = new MPCBETNode(nullptr, BET_ADD_OP, true, 1, false);

  newAddNode->setLeftChild(newMulNode);
  newAddNode->setRightChild(getPowTree(v1, v2, constant, iter + 1));

  return newAddNode;
}

int64_t MPCBET::parseCoefficient(MPCBETNode *betNode) const {
  auto *lc = betNode->getLeftChild();
  auto *rc = betNode->getRightChild();

  if (betNode->isPow()) {
    assert(lc->isLeaf() && "Pow expression should be at bottom of the polynomial tree");
    return (lc->isConstant() ? std::pow(lc->getValue(), rc->getValue()) : 1);
  }
  if (betNode->isConstant() || betNode->isLeaf()) {
    return betNode->getValue();
  }

  return parseCoefficient(lc) * parseCoefficient(rc);
}

int64_t MPCBET::getBinomialCoefficient(int64_t n, int64_t k) {
  auto pascalRow = getPascalRow(n);
  return pascalRow[k];
}

std::vector<int64_t> MPCBET::getPascalRow(int64_t n) {
  if (n >= pascalMatrix.size())
    updatePascalMatrix(n);

  return pascalMatrix[n];
}

void MPCBET::addRoot(Var *newVar, int oldVarId) {
  auto *oldNode = roots[oldVarId]->copy();
  oldNode->setVariable(newVar);
  roots[oldNode->getVariableId()] = oldNode;
}

void MPCBET::addNode(MPCBETNode *betNode) {
  expandNode(betNode);
  addRoot(betNode);
}

void MPCBET::formPolynomials() {
  for (int stopVarId : stopVarIds) {
    auto *polyRoot = roots[stopVarId]->copy();
    do {
      treeAltered = false;
      formPolynomial(polyRoot);
    } while (treeAltered);
    polynomials.push_back(polyRoot);
  }
}

void MPCBET::parseVars(MPCBETNode *betNode) {
  if (betNode->isConstant())
    return;
  if (betNode->isLeaf()) {
    addVar(betNode->getVariableId());
    return;
  }

  parseVars(betNode->getLeftChild());
  parseVars(betNode->getRightChild());
}

MPCBETNode *MPCBET::root() {
  if (!stopVarIds.size())
    return nullptr;

  auto stopVarId = stopVarIds.back();
  auto search = roots.find(stopVarId);
  if (search == roots.end())
    return nullptr;

  return roots[stopVarId];
}

MPCBETNode *MPCBET::polyRoot() const {
  if (!polynomials.size())
    return nullptr;

  return polynomials.back();
}

MPCBETNode *MPCBET::getNextPolyNode() {
  auto *polyNode = polyRoot();

  if (polyNode) {
    polynomials.pop_back();
    return polyNode;
  }

  return nullptr;
}

std::vector<int64_t> MPCBET::extractCoefficents(MPCBETNode *betNode) const {
  std::vector<int64_t> coefficients;
  extractCoefficents(betNode, coefficients);
  return coefficients;
}

std::vector<int64_t> MPCBET::extractExponents(MPCBETNode *betNode) const {
  std::vector<int64_t> exponents;
  extractExponents(betNode, exponents);
  return exponents;
}

std::set<int> MPCBET::extractVars(MPCBETNode *betNode) const {
  std::set<int> varIds;
  extractVars(betNode, varIds);
  return varIds;
}

void MPCBET::escapePows(MPCBETNode *node) {
  if (node->isLeaf())
    return;

  if (!node->isPow()) {
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

  auto *newMulNode = new MPCBETNode(nullptr, BET_MUL_OP, lc, lc, false, 1, false);

  if (rc->getValue() == 1)
    newMulNode->setRightChild(new MPCBETNode(nullptr, 0, true, 1, true));

  for (int i = 0; i < rc->getValue() - 2; ++i)
    newMulNode =
        new MPCBETNode(nullptr, BET_MUL_OP, lc, newMulNode->copy(), false, 1, false);

  node->replace(newMulNode);
}

std::vector<MPCBETNode *> MPCBET::generateFactorizationTrees(int upperLimit = 10) {
  MPCBETNode *root = this->root()->copy();
  escapePows(root);
  reduceAll(root);

  std::vector<MPCBETNode *> factorizations;
  for (int i = 0; i != upperLimit; ++i) {
    factorizations.push_back(root->copy());
    if (!expandLvl(root))
      break;
  }

  return factorizations;
}

bool MPCBET::expandLvl(MPCBETNode *node) {
  // TODO: Add support for operators other than + and *

  if (node->isLeaf())
    return false;

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  if (!(node->isMul() || node->isMatmul()) || !(lc->isAdd() || rc->isAdd())) {
    if (expandLvl(lc))
      return true;
    return expandLvl(rc);
  }

  if (node->isMul())
    expandMul(node);
  if (node->isMatmul())
    expandMatmul(node);
  return true;
}

bool MPCBET::reduceLvl(MPCBETNode *node) {
  // TODO: Add support for operators other than + and *

  if (node->isLeaf())
    return false;

  auto *lc = node->getLeftChild();
  auto *rc = node->getRightChild();
  bool atomic_mul = lc->isMul() && rc->isMul();
  bool matrix_mul = lc->isMatmul() && rc->isMatmul();
  bool children_multiplicative = atomic_mul || matrix_mul;
  bool not_reducible = !node->isAdd() || !children_multiplicative;

  bool reducible = false;
  bool llc_lrc = false;
  bool llc_rrc = false;
  bool rlc_lrc = false;
  bool rlc_rrc = false;

  if (!lc->isLeaf() && !rc->isLeaf()) {
    auto *llc = lc->getLeftChild();
    auto *rlc = lc->getRightChild();
    auto *lrc = rc->getLeftChild();
    auto *rrc = rc->getRightChild();

    llc_lrc = llc->isSameSubTree(lrc);
    llc_rrc = llc->isSameSubTree(rrc);
    rlc_lrc = rlc->isSameSubTree(lrc);
    rlc_rrc = rlc->isSameSubTree(rrc);

    reducible = llc_lrc || llc_rrc || rlc_lrc || rlc_rrc;
  }

  if (not_reducible or !reducible) {
    if (reduceLvl(lc))
      return true;
    return reduceLvl(rc);
  }

  if (atomic_mul)
    collapseMul(node, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc);
  else if (matrix_mul)
    collapseMatmul(node, llc_lrc, llc_rrc, rlc_lrc, rlc_rrc);
  else assert(false);
  
  return true;
}

void MPCBET::expandAll(MPCBETNode *root) {
  while (expandLvl(root))
    ;
}

void MPCBET::reduceAll(MPCBETNode *root) {
  while (reduceLvl(root))
    ;
}

bool isArithmetic(int op) { return op && op < BET_ARITHMETICS_OP_THRESHOLD; }
bool isReveal(int op) { return op == BET_REVEAL_OP; }

/*
 * Auxiliary helpers
 */

int getOperator(CallInstr *callInstr) {
  auto *f = util::getFunc(callInstr->getCallee());
  auto instrName = f->getName();
  if (instrName.find("__add__") != std::string::npos)
    return BET_ADD_OP;
  if (instrName.find("__mul__") != std::string::npos)
    return BET_MUL_OP;
  if (instrName.find("__pow__") != std::string::npos)
    return BET_POW_OP;
  if (instrName.find("matmul") != std::string::npos)
    return BET_MATMUL_OP;
  if (instrName.find("secure_reveal") != std::string::npos)
    return BET_REVEAL_OP;
  return BET_OTHER_OP;
}

types::Type *getTupleType(int n, types::Type *elemType, Module *M) {
  std::vector<types::Type *> tupleTypes;
  for (int i = 0; i != n; ++i)
    tupleTypes.push_back(elemType);
  return M->getTupleType(tupleTypes);
}

/* MPCBET tree construction */

MPCBETNode *MPCParseArithmetic(CallInstr *callInstr) {
  // Arithmetics are binary
  auto *betNode = new MPCBETNode();

  auto op = getOperator(callInstr);
  betNode->setOperator(op);

  auto *lhs = callInstr->front();
  auto *rhs = callInstr->back();
  auto *lhsInstr = cast<CallInstr>(lhs);
  auto *rhsInstr = cast<CallInstr>(rhs);
  auto *lhsConst = cast<IntConst>(lhs);
  auto *rhsConst = cast<IntConst>(rhs);

  if (lhsConst)
    betNode->setLeftChild(
        new MPCBETNode(cast<Var>(lhs), 0, true, lhsConst->getVal(), true));
  else if (!lhsInstr) {
    betNode->setLeftChild(new MPCBETNode(lhs->getUsedVariables().front()));
  } else
    betNode->setLeftChild(MPCParseArithmetic(lhsInstr));

  if (rhsConst)
    betNode->setRightChild(
        new MPCBETNode(cast<Var>(rhs), 0, true, rhsConst->getVal(), true));
  else if (!rhsInstr) {
    betNode->setRightChild(new MPCBETNode(rhs->getUsedVariables().front()));
  } else
    betNode->setRightChild(MPCParseArithmetic(rhsInstr));

  return betNode;
}

void parseInstruction(codon::ir::Value *instruction, MPCBET *bet) {
  auto *retIns = cast<ReturnInstr>(instruction);
  if (retIns) {
    auto vars = retIns->getValue()->getUsedVariables();
    bet->addStopVar(vars.front()->getId());
    return;
  }

  auto *assIns = cast<AssignInstr>(instruction);
  if (!assIns)
    return;

  auto *var = assIns->getLhs();
  auto *callInstr = cast<CallInstr>(assIns->getRhs());
  if (!callInstr)
    return;

  auto op = getOperator(callInstr);
  if (isArithmetic(op)) {
    auto *betNode = MPCParseArithmetic(callInstr);
    betNode->setVariable(var);
    bet->addNode(betNode);
  } else if (isReveal(op)) {
    bet->addRoot(var, util::getVar(callInstr->back())->getId());
    bet->addStopVar(var->getId());
  }
}

MPCBET *parseBET(SeriesFlow *series) {
  auto *bet = new MPCBET();
  for (auto it = series->begin(); it != series->end(); ++it)
    parseInstruction(*it, bet);

  bet->parseVars(bet->root());

  return bet;
}

/* Polynomial MPC optimization */

CallInstr *nextPolynomialCall(CallInstr *v, BodiedFunc *bf, MPCBET *bet) {
  auto polyNode = bet->getNextPolyNode();
  auto coefs = bet->extractCoefficents(polyNode);
  auto exps = bet->extractExponents(polyNode);
  auto vars = bet->extractVars(polyNode);

  auto *M = v->getModule();
  auto *self = M->Nr<VarValue>(bf->arg_front());
  auto *selfType = self->getType();
  auto *funcType = cast<types::FuncType>(bf->getType());
  auto *returnType = funcType->getReturnType();
  auto *inputsType = getTupleType(vars.size(), returnType, M);
  auto *coefsType = getTupleType(coefs.size(), M->getIntType(), M);
  auto *expsType = getTupleType(exps.size(), M->getIntType(), M);

  auto *evalPolyFunc = getOrRealizeSequreInternalMethod(M, "secure_evalp", {selfType, inputsType, coefsType, expsType}, {});
  assert(evalPolyFunc && "secure_evalp not found among Sequre's internal methods");

  std::vector<Value *> inputArgs;
  for (auto it = bf->arg_begin(); it != bf->arg_end(); ++it) {
    if (vars.find((*it)->getId()) == vars.end())
      continue;
    auto *arg = M->Nr<VarValue>(*it);
    inputArgs.push_back(arg);
  }
  std::vector<Value *> coefsArgs;
  for (auto e : coefs)
    coefsArgs.push_back(M->getInt(e));
  std::vector<Value *> expsArgs;
  for (auto e : exps)
    expsArgs.push_back(M->getInt(e));

  auto *inputArg = util::makeTuple(inputArgs, M);
  auto *coefsArg = util::makeTuple(coefsArgs, M);
  auto *expsArg = util::makeTuple(expsArgs, M);

  return util::call(evalPolyFunc, {self, inputArg, coefsArg, expsArg});
}

void convertInstructions(CallInstr *v, BodiedFunc *bf, SeriesFlow *series, MPCBET *bet) {
  auto it = series->begin();
  while (it != series->end()) {
    auto *retIns = cast<ReturnInstr>(*it);
    if (retIns) {
      retIns->setValue(nextPolynomialCall(v, bf, bet));
      ++it;
      continue;
    }

    auto *assIns = cast<AssignInstr>(*it);
    if (!assIns) {
      ++it;
      continue;
    }

    auto *callInstr = cast<CallInstr>(assIns->getRhs());
    if (!callInstr) {
      ++it;
      continue;
    }

    auto op = getOperator(callInstr);
    if (isArithmetic(op)) {
      it = series->erase(it);
      continue;
    }

    if (isReveal(op)) {
      callInstr->setArgs({callInstr->front(), nextPolynomialCall(v, bf, bet)});
      ++it;
      continue;
    }
  }
}

/* Factorization optimizations */

Value *generateExpression(Module *M, MPCBETNode *node) {
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

Value *generateProduct(Module *M, std::vector<Value *> factors,
                       types::Type *returnType) {
  assert(
      factors.size() &&
      "Matrix factorization pass -> Product of shapes: Factors need to be non-empty.");

  if (factors.size() == 1)
    return factors[0];

  auto *mulFunc =
      M->getOrRealizeMethod(returnType, "__mul__", {returnType, returnType});

  std::string const errMsg = "__mul__ not found in type " + returnType->getName();
  assert(mulFunc && errMsg.c_str());

  auto *mulCall = util::call(mulFunc, {factors[0], factors[1]});
  assert(mulCall);

  for (int i = 2; i < factors.size(); ++i)
    mulCall = util::call(mulFunc, {mulCall->getActual(), factors[i]});

  auto *actualMulCall = mulCall->getActual();
  assert(actualMulCall);

  return actualMulCall;
}

std::vector<Value *> parseOpCosts(Module *M, MPCBETNode *node,
                                  std::vector<Value *> &opCosts) {
  if (node->hasAtomicType() || node->isConstant())
    return {M->getInt(1), M->getInt(1)};

  if (node->isLeaf()) {
    auto *shapeMethod =
        M->getOrRealizeMethod(node->getType(), "shape", {node->getType()});
    if (!shapeMethod)
      return {M->getInt(1), M->getInt(1)};

    auto *containerType = util::getReturnType(shapeMethod);
    auto *itemGetterMethod = M->getOrRealizeMethod(containerType, "__getitem__",
                                                   {containerType, M->getIntType()});
    assert(itemGetterMethod);

    auto *shapeCall = util::call(shapeMethod, {node->getVarValue(M)});
    assert(shapeCall);
    auto *getFirstItemCall =
        util::call(itemGetterMethod, {shapeCall->getActual(), M->getInt(0)});
    auto *getSecondItemCall =
        util::call(itemGetterMethod, {shapeCall->getActual(), M->getInt(-1)});
    assert(getFirstItemCall);
    assert(getSecondItemCall);

    return {getFirstItemCall->getActual(), getSecondItemCall->getActual()};
  }

  std::vector<Value *> shape;
  if (node->isElemWiseOp()) {
    shape = parseOpCosts(M, node->getLeftChild(), opCosts);
    assert(shape.size() == 2);
  } else if (node->isMatmul()) {
    auto lshape = parseOpCosts(M, node->getLeftChild(), opCosts);
    auto rshape = parseOpCosts(M, node->getRightChild(), opCosts);
    assert(lshape.size() == 2);
    assert(rshape.size() == 2);
    shape = {lshape[0], lshape[1], rshape[1]};
  } else
    assert(false && "Invalid MPCBET node operation.");

  opCosts.push_back(generateProduct(M, shape, M->getIntType()));
  if (node->isMatmul())
    shape.erase(shape.begin() + 1);

  return shape;
}

Value *generateCostExpression(Module *M, MPCBETNode *factorizationTree) {
  std::vector<Value *> costProducts;
  parseOpCosts(M, factorizationTree, costProducts);

  auto *sumMethod = M->getOrRealizeFunc(
      "sum", {getTupleType(costProducts.size(), M->getIntType(), M)}, {},
      builtinModule);
  assert(sumMethod);

  auto *costProductsTuple = util::makeTuple(costProducts, M);
  auto *callIns = util::call(sumMethod, {costProductsTuple});
  assert(callIns);

  return callIns->getActual();
}

CallInstr *callRouter(Module *M, std::vector<Value *> newVars,
                      std::vector<Value *> costVars) {
  auto *varsTupleType = getTupleType(newVars.size(), newVars[0]->getType(), M);
  auto *costsTupleType = getTupleType(costVars.size(), M->getIntType(), M);
  auto *routerMethod = getOrRealizeSequreInternalMethod(M, "min_cost_router", {varsTupleType, costsTupleType}, {});
  assert(routerMethod);

  auto *varsTuple = util::makeTuple(newVars, M);
  auto *costsTuple = util::makeTuple(costVars, M);
  auto *routerCall = util::call(routerMethod, {varsTuple, costsTuple});
  assert(routerCall);

  return routerCall;
}

void routeFactorizations(CallInstr *v, BodiedFunc *bf, SeriesFlow *series,
                         std::vector<MPCBETNode *> factorizationTrees) {
  auto *M = v->getModule();

  std::vector<Value *> newVars;
  // costVars contains the cost calculation expression for each newVar
  std::vector<Value *> costVars;

  for (auto factorizationTree : factorizationTrees) {
    newVars.push_back(
        util::makeVar(generateExpression(M, factorizationTree), series, bf, true));
    costVars.push_back(
        util::makeVar(generateCostExpression(M, factorizationTree), series, bf, true));
  }

  auto it = series->begin();
  while (it != series->end()) {
    auto *retIns = cast<ReturnInstr>(*it);
    if (!retIns) {
      ++it;
      continue;
    }

    // TODO: Add support for n1 + n2 + ... + nm case
    // TODO: Update type-getter to deduce the return type
    retIns->setValue(callRouter(M, newVars, costVars));
    ++it;
  }
}

/* IR passes */

void applyFactorizationOptimizations(CallInstr *v) {
  auto *f = util::getFunc(v->getCallee());
  if (!isFactOptFunc(f))
    return;

  auto *bf = cast<BodiedFunc>(f);
  auto *series = cast<SeriesFlow>(bf->getBody());

  auto *bet = parseBET(series);
  auto factorizationTrees = bet->generateFactorizationTrees();

  routeFactorizations(v, bf, series, factorizationTrees);
}

void applyPolynomialOptimizations(CallInstr *v) {
  auto *f = util::getFunc(v->getCallee());
  if (!isPolyOptFunc(f) || isSequreFunc(f))
    return;

  auto *bf = cast<BodiedFunc>(f);
  auto *series = cast<SeriesFlow>(bf->getBody());

  auto *bet = parseBET(series);
  bet->formPolynomials();

  convertInstructions(v, bf, series, bet);
}

void MPCOptimizations::handle(CallInstr *v) {
  applyPolynomialOptimizations(v);
  // applyFactorizationOptimizations(v);
}

} // namespace sequre

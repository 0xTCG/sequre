#pragma once

#include "enums.h"
#include "codon/cir/transform/pass.h"
#include "codon/cir/cir.h"
#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;


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
  bool isPow()         const { return operation == power; }
  bool isCommutative() const { return isAdd() || isMul(); }
  bool isExpanded()    const { return expanded; }
  
  bool checkIsVariable()               const { return bool(getVariable()); }
  bool checkIsIntConst()               const { return util::isConst<int64_t>(value); }
  bool checkIsDoubleConst()            const { return util::isConst<double>(value); }
  bool checkIsConst()                  const { return checkIsIntConst() || checkIsDoubleConst(); }
  bool checkIsTypeable()               const { return checkIsVariable() || checkIsConst(); }
  bool checkIsSameTree( BETNode * )    const;
  bool checkIsConsecutiveCommutative() const;
  bool checkIsCipherTensor();
  bool checkIsCiphertext();
  bool checkIsPlaintext();
  
  void swapChildren() { std::swap(leftChild, rightChild); }
  void replace( BETNode * );
  types::Type *getOrRealizeIRType( bool force = false);

  std::string            const getOperationIRName( bool ) const;
  std::string            const getName() const;
  std::string            const getConstStr() const;
  void print( int, int ) const;
};

class BET {
  std::unordered_map<codon::ir::id_t, BETNode *> betPerVar;

public:
  BET() {}
  ~BET() { for ( auto &it: betPerVar ) delete it.second; }

  void addBET( Var* var, BETNode *betNode ) { betPerVar[var->getId()] = betNode; }
  void expandNode( BETNode * );

  bool reduceLvl( BETNode *, bool );
  bool reduceAll( BETNode * );

  bool swapPriorities( BETNode *, BETNode * );
  bool reorderPriority( BETNode * );
  bool reorderPriorities( BETNode * );

  void escapePows( BETNode * );

private:
  std::pair<BETNode *, BETNode *>  findFactorizationNodes( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>& );
  std::pair<BETNode *, BETNode *>  findFactorsInMulTree( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode *, BETNode *, BETNode * );
  BETNode                         *internalIsVisited( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode * );
};

BETNode *parseArithmetic( CallInstr * );
Value   *generateExpression( Module *, BETNode * );

} // namespace sequre

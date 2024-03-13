#pragma once

#include "codon/cir/transform/pass.h"
#include "codon/cir/cir.h"
#include "codon/cir/util/irtools.h"

namespace sequre {

using namespace codon::ir;


class BETNode {
  Value       *value;
  types::Type *irType;
  std::string  operation;
  BETNode     *leftChild;
  BETNode     *rightChild;
  bool         expanded;

public:
  BETNode();
  BETNode( Value *value );
  BETNode( std::string const operation, BETNode *leftChild, BETNode *rightChild );
  BETNode( Value *value, types::Type *irType, std::string const operation, bool expanded );
  ~BETNode() {
    if ( leftChild )  delete leftChild;
    if ( rightChild ) delete rightChild;
  }
  BETNode *copy() const;

  Value             *getValue()       const { return value; };
  Var               *getVariable()    const { return util::getVar(value); }
  VarValue          *getVarValue()    const { return cast<VarValue>(value); };
  BETNode           *getLeftChild()   const { return leftChild; }
  BETNode           *getRightChild()  const { return rightChild; }
  std::string const  getOperation()   const { return operation; }
  int64_t            getIntConst()    const { return util::getConst<int64_t>(value); };
  double             getDoubleConst() const { return util::getConst<double>(value); };
  codon::ir::id_t    getVariableId()  const { return getVariable()->getId(); };
  
  void setValue( Value *value )                    { this->value = value; }
  void setIRType( types::Type *irType )            { this->irType = irType; }
  void setOperation( std::string const operation ) { this->operation = operation; }
  void setLeftChild( BETNode *leftChild )          { this->leftChild = leftChild; }
  void setRightChild( BETNode *rightChild )        { this->rightChild = rightChild; }
  void setExpanded()                               { expanded = true; }

  bool isLeaf()        const { return !leftChild && !rightChild; }
  bool isOperation()   const { return operation != ""; }
  bool isAdd()         const { return operation == Module::ADD_MAGIC_NAME; }
  bool isMul()         const { return operation == Module::MUL_MAGIC_NAME; }
  bool isMatmul()      const { return operation == Module::MATMUL_MAGIC_NAME; }
  bool isPow()         const { return operation == Module::POW_MAGIC_NAME; }
  bool isCommutative() const { return isAdd() || isMul(); }
  bool isExpanded()    const { return expanded; }
  
  bool checkIsVariable()               const { return bool(getVariable()); }
  bool checkIsIntConst()               const { return util::isConst<int64_t>(value); }
  bool checkIsDoubleConst()            const { return util::isConst<double>(value); }
  bool checkIsConst()                  const { return checkIsIntConst() || checkIsDoubleConst(); }
  bool checkIsTypeable()               const { return bool(value->getType()); }
  bool checkIsSameTree( BETNode * )    const;
  bool checkIsConsecutiveCommutative() const;
  bool checkIsSecureContainer();
  bool checkIsCiphertensor();
  bool checkIsCipherCiphertensor();
  bool checkIsPlainCiphertensor();
  
  void swapChildren() { std::swap(leftChild, rightChild); }
  void replace( BETNode * );
  types::Type *getOrRealizeIRType( bool force = false);

  void elementsCount( int &, int & ) const;

  std::string            const getName() const;
  std::string            const getConstStr() const;
  void print( int, int ) const;
};

class BET {
  std::map<codon::ir::id_t, BETNode *> betPerVar;

public:
  static const codon::ir::id_t BET_NO_VAR_ID = -1;
  static const codon::ir::id_t BET_RETURN_ID = -2;

  BET() {}
  ~BET() { for ( auto &it: betPerVar ) delete it.second; }

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = BETNode;
    using pointer           = BETNode*;
    using reference         = BETNode&;

    Iterator( std::vector<BETNode *> recstack ) : recstack(recstack) {}

    reference operator*() const { return *recstack.back(); }
    pointer operator->() { return recstack.back(); }

    Iterator& operator++() {
      if ( !recstack.empty() ) {
        auto cur = recstack.back();
        recstack.pop_back();
        
        auto leftChild  = cur->getLeftChild();
        auto rightChild = cur->getRightChild();
        
        if ( rightChild ) recstack.push_back(rightChild);
        if ( leftChild )  recstack.push_back(leftChild);
      }
      
      return *this;
    }
    Iterator operator++( int ) { Iterator tmp = *this; ++(*this); return tmp; }

    friend bool operator==( const Iterator& a, const Iterator& b ) { return a.recstack == b.recstack; }
    friend bool operator!=( const Iterator& a, const Iterator& b ) { return a.recstack != b.recstack; }

  private:
    std::vector<BETNode *> recstack;
  };
  
  Iterator begin();
  Iterator end() { return Iterator( {} ); }

  void addBET( codon::ir::id_t var_id, BETNode *betNode ) { betPerVar[var_id] = betNode; }
  void expandNode( BETNode * );
  
  BETNode *parseInstruction( Value * );
  void parseSeries( SeriesFlow * );

  bool reduceLvl( BETNode *, bool );
  bool reduceAll( BETNode * );

  bool swapPriorities( BETNode *, BETNode * );
  bool reorderPriority( BETNode * );
  bool reorderPriorities( BETNode * );

  void escapePows( BETNode * );

  std::pair<int, int> elementsCount() const;

  Value *getNodeEncoding( Module *, BETNode *, std::vector<Var *> const & ) const;
  Value *getEncoding( Module *, std::vector<Var *> const & );

private:
  std::pair<BETNode *, BETNode *>  findFactorizationNodes( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>& );
  std::pair<BETNode *, BETNode *>  findFactorsInMulTree( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode *, BETNode *, BETNode * );
  BETNode                         *internalIsVisited( BETNode *, std::vector<BETNode *>&, std::unordered_map<BETNode *, std::vector<BETNode *>>&, BETNode * );
};

BETNode *parseBinaryArithmetic( CallInstr * );
Value   *generateExpression( Module *, BETNode * );

} // namespace sequre

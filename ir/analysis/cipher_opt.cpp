#include "cipher_opt.h"
#include "../helpers/utils.h"

#include "codon/cir/util/irtools.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// How @mhe_cipher_opt plans a rewrite (rules and the cases behind them: the cluster 24 handoff, "Design (as built)").
//
// Operators. A call is an operator node when it calls `__add__` or `__mul__` of its first argument's type with two
// operands of types the planner knows (int, float, ndarray, Sharetensor, Ciphertensor[Ciphertext], Ciphertensor
// [Plaintext], MPP, MPA, MPU), and its result is secure exactly when an operand is. Reflected forms reach the IR as
// statement-expressions with temporaries, so they are atoms (below).
//
// Atoms. Everything else under an operator is an atom: a variable, an int or float literal, a field read of a
// variable (self.w), x.T of a tensor and x @ y of tensors that are plain atoms are the plain atoms; any other value
// makes its expression stay as written (the planner still looks for expressions inside it), and so does a sum or
// product of two public operands of integer type (it wraps at 64 bits as written). Plain atoms have no effects and
// nothing in a planned expression can change them, so an occurrence may move or merge; two occurrences are the same
// atom when they read the same variable (field, x.T, x @ y) or are equal literals.
//
// Folding. A top-level `v = <expr>` folds into the top-level statement that uses it (`x = <e>`, `return <e>` or a
// bare `<e>`) only if v is a local, non-parameter variable assigned once in the whole function, never addressed,
// every read of v is in that expression (or in a definition folded with it), only such plain assignments lie
// between the two statements, and none of them assigns a variable the definition reads. The assignment is then
// removed: the pass emptied it of readers. Nothing else is ever deleted.
//
// Factoring. On sums of products of atoms (repeated subexpressions are separate occurrences, never shared nodes), a
// factor in two or more terms is taken out (the most frequent first, secure before public); the cofactors are summed
// and factored in turn. Public operands of a sum or a product are combined first, and the rest pairwise, the least
// deep first. Two public operands of integer type are combined only when both are literals and the result fits in 64
// bits (Sequre's arithmetic on a secure operand does not wrap), and a combination with a float must give a float. Next
// to Sharetensors only literals are combined, and only while the value stays within the 31 integer bits Sequre
// encodes a public value in.
//
// Typing. Each new operation is realized by Codon: l.__op__(r), else r.__rop__(l), else the swapped operands (both
// operators commute on these types); under @sequre an operation the expressiveness pass will rewrite must also realize
// as Internal.secure_add / secure_mul. The new expression must have the original type.
//
// Run-time state. A Sharetensor's fixed-point flag is not in its type: Sequre refuses to add a fixed-point and an
// integer Sharetensor, and adds a float to an integer one wrongly. A plan's additions run only when each adds two of a
// kind and a float only to a fixed-point Sharetensor (the run-time helper fp_agree; a product is fixed point when a
// factor is, a float makes it so); otherwise the expression as written runs, its folded definitions' assignments first.
// Ciphertensors and MP tensors carry storage the types do not show (encodings, transposition, partitioning, the
// aggregate), and their products refresh operands in place: an expression with two or more distinct secure tensors of
// these kinds stays as written, leaving storage to a layout planner (BOSTAN: Shechi, section 8.2).
//
// Levels. An MP value times an int literal below MHE_MUL_TO_ADD_THRESHOLD is additions: no level.
//
// Cost. The plan replaces the original only if its (secure `@` products, secure-by-secure products, secure-by-public
// products, secure additions) is lexicographically smaller, it evaluates no `@` product more often than the original
// does, and no secure atom ends up under more secure products than in the original (depth is levels, levels are
// bootstraps).
//
// `@` chains of three or more Sharetensors, each a plain expression, go to the run-time helper matmul_chain with the
// order as written, which it keeps unless Sharetensor's shape cost model makes another order strictly cheaper (only
// that model can cost a whole chain before its products exist: chains of other types stay as written).

namespace sequre {

using namespace codon::ir;

namespace {

using IrId = codon::ir::id_t;

constexpr int NONE = INT_MIN;

enum class Op { ADD, MUL };

std::string magicOf( Op op ) { return op == Op::ADD ? Module::ADD_MAGIC_NAME : Module::MUL_MAGIC_NAME; }
std::string reflectedMagicOf( Op op ) { return op == Op::ADD ? "__radd__" : "__rmul__"; }

enum class AtomKind { VAR, INT, FLOAT, FIELD, TRANSPOSE, MATMUL };

struct Atom {
  AtomKind kind = AtomKind::VAR;
  std::string key;
  types::Type *type = nullptr;
  bool secret = false;
  Var *var = nullptr;
  int64_t ival = 0;
  double fval = 0.0;
  std::string field;
  Func *func = nullptr;
  int inner = -1;
  int right = -1; // MATMUL: the right operand (inner is the left one)
};

// A planned expression: sums and products of atoms by value (no node is shared)
struct Expr {
  enum Kind { ATOM, SUM, PROD };
  Kind kind = ATOM;
  int atom = -1;
  std::vector<Expr> kids;
  std::string key;     // structural identity, set by normalize
  bool secret = false; // set by normalize
};

struct Cost {
  int mm = 0, ss = 0, sp = 0, sa = 0;
  bool operator<( const Cost &o ) const { return std::tie(mm, ss, sp, sa) < std::tie(o.mm, o.ss, o.sp, o.sa); }
  bool operator==( const Cost &o ) const { return std::tie(mm, ss, sp, sa) == std::tie(o.mm, o.ss, o.sp, o.sa); }
  Cost operator+( const Cost &o ) const { return Cost{mm + o.mm, ss + o.ss, sp + o.sp, sa + o.sa}; }
  Cost operator-( const Cost &o ) const { return Cost{mm - o.mm, ss - o.ss, sp - o.sp, sa - o.sa}; }
  std::string str() const {
    return "(" + std::to_string(mm) + " secure matmuls, " + std::to_string(ss) + " secure products, " +
           std::to_string(sp) + " by public, " + std::to_string(sa) + " secure additions)";
  }
};

// How often each `@` product (by atom key) is evaluated
using MatmulCounts = std::map<std::string, int>;

// MPP and MPA multiply by an int of smaller absolute value with additions, which cost no level
// (stdlib/sequre/constants.codon: MHE_MUL_TO_ADD_THRESHOLD)
constexpr int64_t MP_ADD_CHAIN_THRESHOLD = 7;

bool isMPKind( OperandKind k ) { return k == OperandKind::MPP || k == OperandKind::MPA || k == OperandKind::MPU; }

// A node of an emitted (binary, typed) expression
struct ENode {
  bool isAtom = false;
  int atom = -1;
  Op op = Op::ADD;
  int l = -1, r = -1;     // the operation is l op r
  bool reflected = false; // called as r.__rop__(l)
  Func *func = nullptr;
  types::Type *type = nullptr;
  bool secret = false;
  int need = NONE;        // max over secure atoms below: secure products above it here minus in the original
  std::optional<int64_t> ival; // an int computed from literals only: its value
  std::optional<double> lit;   // a number computed from literals only: its value
};

struct Realized {
  Func *func = nullptr;
  types::Type *ret = nullptr;
};

// A run-time condition of a plan: helper((bases...), (ints...)...) -> bool, the bases being atoms read again
struct Check {
  Func *helper = nullptr;
  Func *basesTuple = nullptr;
  std::vector<int> bases;
  std::vector<std::pair<Func *, std::vector<int64_t>>> ints; // each int tuple after the bases: its __new__, values
};

struct Rewrite {
  Value *parent;
  Value *oldValue;
  int root;
  codon::SrcInfo info;
  std::vector<Check> checks;    // all must hold for the plan to run; the expression as written runs otherwise
  std::vector<Value *> folded;  // the folded definitions' assignments, in statement order
};

struct Chain {
  Value *parent;
  Value *root;
  Func *helper;
  Func *tupleNew;
  Func *ordersNew;
};

using SeriesIt = std::list<Value *>::iterator;

struct Ctx {
  Module *M = nullptr;
  BodiedFunc *f = nullptr;
  SeriesFlow *body = nullptr;
  Var *mpcVar = nullptr;
  types::Type *mpcType = nullptr;
  bool sequre = false;

  std::vector<Atom> atoms;
  std::unordered_map<std::string, int> atomIndex;

  std::unordered_map<IrId, int> assigns, reads, pointers;
  std::unordered_set<IrId> params;

  std::set<std::string> pairs; // (operator, left type, right type) of every operator node the function holds
  std::vector<ENode> pool;

  std::vector<Rewrite> rewrites;
  std::vector<Chain> chains;
  std::vector<SeriesIt> erasures;

  // while emitting a plan: whether the expression holds Sharetensors
  bool sharetensorPlan = false;
};

int countOf( const std::unordered_map<IrId, int> &m, IrId id ) {
  auto it = m.find(id);
  return it == m.end() ? 0 : it->second;
}

// ---------------------------------------------------------------- variables

void collectVarInfo( Ctx &c, Value *v ) {
  if ( !v ) return;
  if ( auto *vv = cast<VarValue>(v) ) {
    auto *var = vv->getVar();
    if ( var && !isA<Func>(var) ) c.reads[var->getId()]++;
    return;
  }
  if ( auto *pv = cast<PointerValue>(v) ) {
    if ( auto *var = pv->getVar() ) c.pointers[var->getId()]++;
    return;
  }
  // The variables a node binds: an assignment's target, a loop's or a handler's variable. Anything else that names a
  // variable counts as binding it too, which only keeps that variable from being folded.
  for ( auto *var : v->getUsedVariables() )
    if ( var && !isA<Func>(var) ) c.assigns[var->getId()]++;
  for ( auto *u : v->getUsedValues() ) collectVarInfo(c, u);
}

// ---------------------------------------------------------------- atoms and operators

int addAtom( Ctx &c, Atom a ) {
  auto it = c.atomIndex.find(a.key);
  if ( it != c.atomIndex.end() ) return it->second;
  std::string key = a.key;
  c.atoms.push_back(std::move(a));
  int idx = int(c.atoms.size()) - 1;
  c.atomIndex[key] = idx;
  return idx;
}

std::string doubleKey( double d ) {
  uint64_t bits = 0;
  std::memcpy(&bits, &d, sizeof bits);
  std::ostringstream ss;
  ss << std::hex << bits;
  return ss.str();
}

// fn is a method of the first argument's type taking exactly these argument types
bool isMethodOfFirst( Func *fn, const std::vector<types::Type *> &args ) {
  if ( !fn || args.empty() || !hasParamTypes(fn, args) ) return false;
  if ( auto *pt = fn->getParentType() ) return pt->is(args[0]);
  return true;
}

int parseAtom( Ctx &c, Value *v, bool arithmetic );
bool asMatmul( Value *v, Value *&l, Value *&r );

int parseField( Ctx &c, ExtractInstr *ex, Atom a ) {
  int inner = parseAtom(c, ex->getVal(), false);
  if ( inner < 0 ) return -1;
  a.kind = AtomKind::FIELD;
  a.inner = inner;
  a.field = ex->getField();
  a.key = "x(" + c.atoms[inner].key + ")." + a.field;
  return addAtom(c, std::move(a));
}

// An atom of a planned expression (arithmetic: an operand of + or *, whose type the planner must know), or the
// object of a field read (any variable or field read). Returns its index, or -1 when the value is no plain atom.
int parseAtom( Ctx &c, Value *v, bool arithmetic ) {
  if ( !v || !v->getType() ) return -1;
  auto *t = v->getType();
  auto k = operandKind(t);
  if ( arithmetic && k == OperandKind::UNKNOWN ) return -1;
  Atom a;
  a.type = t;
  a.secret = isSecretKind(k);
  if ( auto *vv = cast<VarValue>(v) ) {
    auto *var = vv->getVar();
    if ( !var || isA<Func>(var) ) return -1;
    a.kind = AtomKind::VAR;
    a.var = var;
    a.key = "v" + std::to_string(var->getId());
    return addAtom(c, std::move(a));
  }
  if ( auto *ex = cast<ExtractInstr>(v) ) return parseField(c, ex, std::move(a));
  if ( !arithmetic ) return -1;
  if ( k == OperandKind::INT && util::isConst<int64_t>(v) ) {
    a.kind = AtomKind::INT;
    a.ival = util::getConst<int64_t>(v);
    a.key = "i" + std::to_string(a.ival) + ":" + t->getName();
    return addAtom(c, std::move(a));
  }
  if ( k == OperandKind::FLOAT && util::isConst<double>(v) ) {
    a.kind = AtomKind::FLOAT;
    a.fval = util::getConst<double>(v);
    a.key = "f" + doubleKey(a.fval) + ":" + t->getName();
    return addAtom(c, std::move(a));
  }
  if ( auto *call = cast<CallInstr>(v) ) {
    // x @ y of tensors: the product itself is no operator node (the planner does not factor it), and like the
    // operators it changes neither operand
    Value *l = nullptr, *r = nullptr;
    if ( isTensorKind(k) && asMatmul(v, l, r) && isTensorKind(operandKind(l->getType())) &&
         isTensorKind(operandKind(r->getType())) ) {
      auto *fn = util::getFunc(call->getCallee());
      auto *ret = returnTypeOf(fn);
      int li = ret && ret->is(t) ? parseAtom(c, l, true) : -1;
      int ri = li >= 0 ? parseAtom(c, r, true) : -1;
      if ( ri < 0 ) return -1;
      a.kind = AtomKind::MATMUL;
      a.func = fn;
      a.inner = li;
      a.right = ri;
      a.key = "M" + std::to_string(fn->getId()) + "(" + c.atoms[li].key + "," + c.atoms[ri].key + ")";
      return addAtom(c, std::move(a));
    }
    // x.T of a tensor: a property that builds a new object from x and changes nothing
    auto *fn = util::getFunc(call->getCallee());
    if ( fn && call->numArgs() == 1 && fn->getUnmangledName() == "T" && isTensorKind(k) ) {
      auto *arg = call->front();
      if ( arg && arg->getType() && isTensorKind(operandKind(arg->getType())) &&
           isMethodOfFirst(fn, {arg->getType()}) && returnTypeOf(fn) && returnTypeOf(fn)->is(t) ) {
        int inner = parseAtom(c, arg, true);
        if ( inner >= 0 ) {
          a.kind = AtomKind::TRANSPOSE;
          a.func = fn;
          a.inner = inner;
          a.key = "T" + std::to_string(fn->getId()) + "(" + c.atoms[inner].key + ")";
          return addAtom(c, std::move(a));
        }
      }
    }
  }
  return -1;
}

bool asOperator( Value *v, Op &op, Value *&l, Value *&r ) {
  auto *call = cast<CallInstr>(v);
  if ( !call || call->numArgs() != 2 ) return false;
  auto *fn = util::getFunc(call->getCallee());
  if ( !fn ) return false;
  auto name = fn->getUnmangledName();
  if ( name == Module::ADD_MAGIC_NAME ) op = Op::ADD;
  else if ( name == Module::MUL_MAGIC_NAME ) op = Op::MUL;
  else return false;
  l = call->front();
  r = call->back();
  if ( !l || !r || !l->getType() || !r->getType() || !call->getType() ) return false;
  auto lk = operandKind(l->getType()), rk = operandKind(r->getType()), k = operandKind(call->getType());
  if ( lk == OperandKind::UNKNOWN || rk == OperandKind::UNKNOWN || k == OperandKind::UNKNOWN ) return false;
  if ( isSecretKind(k) != (isSecretKind(lk) || isSecretKind(rk)) ) return false;
  auto *ret = returnTypeOf(fn);
  return isMethodOfFirst(fn, {l->getType(), r->getType()}) && ret && ret->is(call->getType());
}

bool asMatmul( Value *v, Value *&l, Value *&r ) {
  auto *call = cast<CallInstr>(v);
  if ( !call || call->numArgs() != 2 ) return false;
  auto *fn = util::getFunc(call->getCallee());
  if ( !fn || fn->getUnmangledName() != Module::MATMUL_MAGIC_NAME ) return false;
  l = call->front();
  r = call->back();
  if ( !l || !r || !l->getType() || !r->getType() ) return false;
  return isMethodOfFirst(fn, {l->getType(), r->getType()});
}

// The expression as written: operators over plain atoms only. A sum or product of two public operands of integer
// type is no such expression: it wraps at 64 bits as written, which a regrouped expression would not.
bool parseExpr( Ctx &c, Value *v, Expr &out ) {
  Op op = Op::ADD;
  Value *l = nullptr, *r = nullptr;
  if ( asOperator(v, op, l, r) ) {
    if ( !isSecretKind(operandKind(l->getType())) && !isSecretKind(operandKind(r->getType())) &&
         !isFloatPublic(l->getType()) && !isFloatPublic(r->getType()) )
      return false;
    Expr le, re;
    if ( !parseExpr(c, l, le) || !parseExpr(c, r, re) ) return false;
    out = Expr();
    out.kind = op == Op::ADD ? Expr::SUM : Expr::PROD;
    out.kids.push_back(std::move(le));
    out.kids.push_back(std::move(re));
    return true;
  }
  int a = parseAtom(c, v, true);
  if ( a < 0 ) return false;
  out = Expr();
  out.kind = Expr::ATOM;
  out.atom = a;
  return true;
}

void atomVarReads( Ctx &c, int a, std::map<IrId, int> &out ) {
  const Atom &at = c.atoms[a];
  if ( at.kind == AtomKind::VAR ) out[at.var->getId()]++;
  if ( at.inner >= 0 ) atomVarReads(c, at.inner, out);
  if ( at.right >= 0 ) atomVarReads(c, at.right, out);
}

// Variables an expression reads: all of them, or only those it reads as whole atoms (the ones folding can replace)
void varReads( Ctx &c, const Expr &e, std::map<IrId, int> &out, bool directOnly ) {
  if ( e.kind == Expr::ATOM ) {
    if ( !directOnly ) atomVarReads(c, e.atom, out);
    else if ( c.atoms[e.atom].kind == AtomKind::VAR ) out[c.atoms[e.atom].var->getId()]++;
    return;
  }
  for ( auto &k : e.kids ) varReads(c, k, out, directOnly);
}

// ---------------------------------------------------------------- the expression as written

// The `@` products one occurrence of an atom evaluates (an x @ y atom and those of its operands): each by key into
// counts, the secure ones into secure
void atomMatmuls( Ctx &c, int a, MatmulCounts *counts, int *secure ) {
  const Atom &at = c.atoms[a];
  if ( at.kind == AtomKind::MATMUL ) {
    if ( counts ) (*counts)[at.key]++;
    if ( secure && at.secret ) (*secure)++;
  }
  if ( at.inner >= 0 ) atomMatmuls(c, at.inner, counts, secure);
  if ( at.right >= 0 ) atomMatmuls(c, at.right, counts, secure);
}

void exprMatmuls( Ctx &c, const Expr &e, MatmulCounts &counts ) {
  if ( e.kind == Expr::ATOM ) {
    atomMatmuls(c, e.atom, &counts, nullptr);
    return;
  }
  for ( auto &k : e.kids ) exprMatmuls(c, k, counts);
}

bool secretOf( Ctx &c, const Expr &e ) {
  if ( e.kind == Expr::ATOM ) return c.atoms[e.atom].secret;
  for ( auto &k : e.kids )
    if ( secretOf(c, k) ) return true;
  return false;
}

// A secure value of an MP kind (its products keep the kind)
bool mpSecretOf( Ctx &c, const Expr &e ) {
  if ( e.kind == Expr::ATOM ) return c.atoms[e.atom].secret && isMPKind(operandKind(c.atoms[e.atom].type));
  for ( auto &k : e.kids )
    if ( mpSecretOf(c, k) ) return true;
  return false;
}

bool smallIntLiteral( Ctx &c, const Expr &e ) {
  if ( e.kind != Expr::ATOM ) return false;
  const Atom &a = c.atoms[e.atom];
  return a.kind == AtomKind::INT && a.ival > -MP_ADD_CHAIN_THRESHOLD && a.ival < MP_ADD_CHAIN_THRESHOLD;
}

// Cost of the tree as written (a node of more kids evaluates left to right) and, for each secure atom, the most secure
// products above one of its occurrences
void writtenCost( Ctx &c, const Expr &e, int above, Cost &cost, std::map<std::string, int> &paths ) {
  if ( e.kind == Expr::ATOM ) {
    const Atom &a = c.atoms[e.atom];
    atomMatmuls(c, e.atom, nullptr, &cost.mm);
    if ( a.secret ) {
      auto it = paths.find(a.key);
      if ( it == paths.end() || it->second < above ) paths[a.key] = above;
    }
    return;
  }
  size_t n = e.kids.size();
  std::vector<int> counted(n, 0);
  bool acc = secretOf(c, e.kids[0]);
  bool accMP = mpSecretOf(c, e.kids[0]);
  for ( size_t i = 1; i < n; ++i ) {
    bool s = secretOf(c, e.kids[i]);
    if ( e.kind == Expr::PROD ) {
      if ( acc && s ) cost.ss++;
      else if ( acc || s ) cost.sp++;
      counted[i] = (acc || s) ? 1 : 0;
      // (an MP value times a small int literal: additions, no level)
      if ( (accMP && !s && smallIntLiteral(c, e.kids[i])) ||
           (i == 1 && !acc && smallIntLiteral(c, e.kids[0]) && mpSecretOf(c, e.kids[i])) )
        counted[i] = 0;
      accMP = accMP || mpSecretOf(c, e.kids[i]);
    } else if ( acc || s ) {
      cost.sa++;
    }
    acc = acc || s;
  }
  for ( size_t i = 0; i < n; ++i ) {
    int d = 0;
    for ( size_t j = std::max<size_t>(i, 1); j < n; ++j ) d += counted[j];
    writtenCost(c, e.kids[i], above + d, cost, paths);
  }
}

// ---------------------------------------------------------------- normalizing and factoring

void normalize( Ctx &c, Expr &e ) {
  if ( e.kind == Expr::ATOM ) {
    e.key = c.atoms[e.atom].key;
    e.secret = c.atoms[e.atom].secret;
    return;
  }
  std::vector<Expr> flat;
  for ( auto &k : e.kids ) {
    normalize(c, k);
    if ( k.kind == e.kind )
      for ( auto &g : k.kids ) flat.push_back(std::move(g));
    else
      flat.push_back(std::move(k));
  }
  std::stable_sort(flat.begin(), flat.end(), []( const Expr &a, const Expr &b ) { return a.key < b.key; });
  if ( flat.size() == 1 ) {
    Expr only = std::move(flat[0]);
    e = std::move(only);
    return;
  }
  e.kids = std::move(flat);
  e.secret = false;
  std::string key = e.kind == Expr::SUM ? "S(" : "P(";
  for ( size_t i = 0; i < e.kids.size(); ++i ) {
    if ( i ) key += ",";
    key += e.kids[i].key;
    e.secret = e.secret || e.kids[i].secret;
  }
  e.key = key + ")";
}

std::vector<const Expr *> factorsOf( const Expr &t ) {
  std::vector<const Expr *> fs;
  if ( t.kind == Expr::PROD )
    for ( auto &k : t.kids ) fs.push_back(&k);
  else
    fs.push_back(&t);
  return fs;
}

void factor( Ctx &c, Expr &e );

// e: a normalized sum. Takes a factor shared by two or more terms out of them, until no factor is shared.
void factorSum( Ctx &c, Expr &e ) {
  while ( e.kind == Expr::SUM ) {
    std::map<std::string, int> count;
    std::map<std::string, bool> secret;
    for ( auto &t : e.kids ) {
      std::set<std::string> seen;
      for ( auto *f : factorsOf(t) )
        if ( seen.insert(f->key).second ) {
          count[f->key]++;
          secret[f->key] = f->secret;
        }
    }
    std::vector<std::string> cands;
    for ( auto &kv : count )
      if ( kv.second >= 2 ) cands.push_back(kv.first);
    std::sort(cands.begin(), cands.end(), [&]( const std::string &a, const std::string &b ) {
      if ( count[a] != count[b] ) return count[a] > count[b];
      if ( secret[a] != secret[b] ) return secret[a];
      return a < b;
    });
    bool done = false;
    for ( auto &cand : cands ) {
      // The terms holding the factor beside another one (taking it out of a term that is only the factor would leave a
      // 1 behind, which the planner does not write)
      std::vector<bool> inGroup(e.kids.size(), false);
      size_t groupSize = 0;
      for ( size_t i = 0; i < e.kids.size(); ++i ) {
        auto fs = factorsOf(e.kids[i]);
        if ( fs.size() < 2 ) continue;
        for ( auto *f : fs )
          if ( f->key == cand ) {
            inGroup[i] = true;
            ++groupSize;
            break;
          }
      }
      if ( groupSize < 2 ) continue;
      Expr factorExpr;
      Expr inner;
      inner.kind = Expr::SUM;
      std::vector<Expr> rest;
      for ( size_t i = 0; i < e.kids.size(); ++i ) {
        if ( !inGroup[i] ) {
          rest.push_back(std::move(e.kids[i]));
          continue;
        }
        Expr cof;
        cof.kind = Expr::PROD;
        bool removed = false;
        for ( auto &k : e.kids[i].kids ) {
          if ( !removed && k.key == cand ) {
            if ( factorExpr.key.empty() ) factorExpr = k;
            removed = true;
            continue;
          }
          cof.kids.push_back(std::move(k));
        }
        normalize(c, cof);
        inner.kids.push_back(std::move(cof));
      }
      normalize(c, inner);
      factor(c, inner);
      Expr term;
      term.kind = Expr::PROD;
      term.kids.push_back(std::move(factorExpr));
      term.kids.push_back(std::move(inner));
      normalize(c, term);
      rest.push_back(std::move(term));
      e.kids = std::move(rest);
      normalize(c, e);
      done = true;
      break;
    }
    if ( !done ) break;
  }
}

void factor( Ctx &c, Expr &e ) {
  if ( e.kind == Expr::ATOM ) return;
  for ( auto &k : e.kids ) factor(c, k);
  normalize(c, e);
  if ( e.kind == Expr::SUM ) factorSum(c, e);
}

size_t exprSize( const Expr &e ) {
  size_t n = 1;
  for ( auto &k : e.kids ) n += exprSize(k);
  return n;
}

// Folding copies a definition into each of its reads: a chain of definitions read twice each doubles at every step
constexpr size_t MAX_FOLDED_SIZE = 512;

// Replaces each read of a variable in subst by (a copy of) its definition, one level
void substituteOnce( Ctx &c, Expr &e, const std::unordered_map<IrId, const Expr *> &subst ) {
  if ( e.kind == Expr::ATOM ) {
    const Atom &a = c.atoms[e.atom];
    if ( a.kind != AtomKind::VAR ) return;
    auto it = subst.find(a.var->getId());
    if ( it != subst.end() ) e = *it->second;
    return;
  }
  for ( auto &k : e.kids ) substituteOnce(c, k, subst);
}

// ---------------------------------------------------------------- typing and emitting

std::string pairKey( Op op, types::Type *lt, types::Type *rt ) {
  return std::string(op == Op::ADD ? "+" : "*") + lt->getName() + "|" + rt->getName();
}

void collectPairs( Ctx &c, Value *v ) {
  if ( !v ) return;
  Op op = Op::ADD;
  Value *l = nullptr, *r = nullptr;
  if ( asOperator(v, op, l, r) ) c.pairs.insert(pairKey(op, l->getType(), r->getType()));
  for ( auto *u : v->getUsedValues() ) collectPairs(c, u);
}

// Operand pairings a new operation may make beyond what typing checks. Sequre raises at run time when a
// Ciphertensor[Plaintext] meets anything but a Ciphertensor[Ciphertext]; and two secure operands of different types
// (an MPP and an MPA, Sharetensors of different shapes) are paired only the way the function already pairs them.
bool pairAllowed( Ctx &c, Op op, types::Type *lt, types::Type *rt ) {
  auto lk = operandKind(lt), rk = operandKind(rt);
  if ( (lk == OperandKind::PLAIN_CT && rk != OperandKind::CIPHER_CT) ||
       (rk == OperandKind::PLAIN_CT && lk != OperandKind::CIPHER_CT) )
    return false;
  if ( isSecretKind(lk) && isSecretKind(rk) && !lt->is(rt) )
    return c.pairs.count(pairKey(op, lt, rt)) || c.pairs.count(pairKey(op, rt, lt));
  return true;
}

// ir/expr.cpp rewrites an __add__ or __mul__ of a @sequre function into Internal.secure_add / secure_mul(mpc, l, r)
// when its result is a Sharetensor or a Ciphertensor, except a Sharetensor product with an int operand
bool rewrittenBySequre( Ctx &c, Op op, types::Type *lt, types::Type *rt, types::Type *ret ) {
  if ( !isSharetensor(ret) && !isCiphertensor(ret) ) return false;
  if ( op == Op::MUL && isSharetensor(ret) ) {
    auto *intType = c.M->getIntType();
    if ( lt->is(intType) || rt->is(intType) ) return false;
  }
  return true;
}

// Realizations for the whole module (a failed one costs a type check each time it is tried again)
struct RealizeCache {
  Module *M = nullptr;
  std::unordered_map<std::string, std::optional<Realized>> map;
};

RealizeCache &realizeCache( Module *M ) {
  static RealizeCache cache;
  if ( cache.M != M ) {
    cache.M = M;
    cache.map.clear();
  }
  return cache;
}

std::optional<Realized> realizeOp( Ctx &c, Op op, types::Type *lt, types::Type *rt, bool reflected ) {
  if ( !pairAllowed(c, op, lt, rt) ) return {};
  std::string key = std::string(reflected ? "r" : "f") + (op == Op::ADD ? "+" : "*") + lt->getName() + "|" +
                    rt->getName() + (c.sequre ? "|sequre|" + c.mpcType->getName() : "");
  auto &cache = realizeCache(c.M).map;
  auto it = cache.find(key);
  if ( it != cache.end() ) return it->second;

  std::optional<Realized> res;
  auto *self = reflected ? rt : lt;
  auto *other = reflected ? lt : rt;
  Func *fn = realizeMethodQuiet(c.M, self, reflected ? reflectedMagicOf(op) : magicOf(op), {self, other});
  auto *ret = returnTypeOf(fn);
  if ( fn && ret && (!fn->getParentType() || fn->getParentType()->is(self)) ) {
    auto lk = operandKind(lt), rk = operandKind(rt), k = operandKind(ret);
    bool ok = k != OperandKind::UNKNOWN && isSecretKind(k) == (isSecretKind(lk) || isSecretKind(rk));
    if ( ok && c.sequre && !reflected && rewrittenBySequre(c, op, lt, rt, ret) ) {
      Func *s = c.mpcType ? realizeInternalMethodQuiet(c.M, op == Op::ADD ? "secure_add" : "secure_mul",
                                                       {c.mpcType, lt, rt})
                          : nullptr;
      ok = s && returnTypeOf(s) && returnTypeOf(s)->is(ret);
    }
    if ( ok ) res = Realized{fn, ret};
  }
  cache[key] = res;
  return res;
}

int combine( Ctx &c, Op op, int x, int y ) {
  // A secure operand on the left first: Sequre's types define the operators on themselves, and int, float and ndarray
  // carry overloads for a secure right operand
  std::vector<std::pair<int, int>> orders = {{x, y}, {y, x}};
  if ( !c.pool[x].secret && c.pool[y].secret ) std::swap(orders[0], orders[1]);
  for ( auto &lr : orders ) {
    int l = lr.first, r = lr.second;
    for ( bool refl : {false, true} ) {
      auto real = realizeOp(c, op, c.pool[l].type, c.pool[r].type, refl);
      if ( !real ) continue;
      ENode n;
      n.op = op;
      n.l = l;
      n.r = r;
      n.reflected = refl;
      n.func = real->func;
      n.type = real->ret;
      n.secret = c.pool[l].secret || c.pool[r].secret;
      int m = std::max(c.pool[l].need, c.pool[r].need);
      // (an MP value times a small int literal: additions, no level)
      auto smallInt = [&]( int x ) {
        return !c.pool[x].secret && c.pool[x].ival && *c.pool[x].ival > -MP_ADD_CHAIN_THRESHOLD &&
               *c.pool[x].ival < MP_ADD_CHAIN_THRESHOLD;
      };
      auto mp = [&]( int x ) { return c.pool[x].secret && isMPKind(operandKind(c.pool[x].type)); };
      bool levelFree = op == Op::MUL && ((mp(l) && smallInt(r)) || (mp(r) && smallInt(l)));
      n.need = (m != NONE && op == Op::MUL && n.secret && !levelFree) ? m + 1 : m;
      c.pool.push_back(n);
      return int(c.pool.size()) - 1;
    }
  }
  return -1;
}

// The public operands of a sum or product, combined where that cannot wrap: floats (and ndarrays of floats) with one
// another and with integer ones when the result is a float, two integer ones only when both are computed from
// literals and the result fits in 64 bits. Returns the public operands left, or nothing when a combination of floats
// does not type as one.
std::optional<std::vector<int>> combinePublics( Ctx &c, Op op, const std::vector<int> &pub ) {
  if ( c.sharetensorPlan ) {
    // Next to a fixed-point Sharetensor, Sequre encodes a public value in 31 integer bits (double_to_fp saturates
    // beyond) and 32 fractional ones: literals only, and only while the combined value stays within them (no closer to
    // 0 than 2^-16, where the encoding has lost half its precision, unless it is 0)
    std::vector<int> out;
    int acc = -1;
    for ( int p : pub ) {
      if ( !c.pool[p].lit ) {
        out.push_back(p);
        continue;
      }
      if ( acc < 0 ) {
        acc = p;
        continue;
      }
      double v = op == Op::ADD ? *c.pool[acc].lit + *c.pool[p].lit : *c.pool[acc].lit * *c.pool[p].lit;
      bool fits = std::fabs(v) < 2147483648.0 && (v == 0.0 || std::fabs(v) >= 1.0 / 65536.0);
      int n = fits ? combine(c, op, acc, p) : -1;
      if ( n < 0 ) {
        out.push_back(p);
        continue;
      }
      c.pool[n].lit = v;
      acc = n;
    }
    if ( acc >= 0 ) out.push_back(acc);
    return out;
  }
  std::vector<int> floats, ints, out;
  for ( int p : pub ) (isFloatPublic(c.pool[p].type) ? floats : ints).push_back(p);
  int acc = -1;
  for ( int p : floats ) {
    if ( acc < 0 ) {
      acc = p;
      continue;
    }
    int n = combine(c, op, acc, p);
    if ( n < 0 || !isFloatPublic(c.pool[n].type) ) return {};
    acc = n;
  }
  std::vector<int> rest;
  for ( int p : ints ) {
    int n = acc >= 0 ? combine(c, op, acc, p) : -1;
    if ( n >= 0 && isFloatPublic(c.pool[n].type) ) acc = n;
    else rest.push_back(p);
  }
  int lit = -1;
  for ( int p : rest ) {
    if ( !c.pool[p].ival ) {
      out.push_back(p);
      continue;
    }
    if ( lit < 0 ) {
      lit = p;
      continue;
    }
    int64_t x = *c.pool[lit].ival, y = *c.pool[p].ival, v = 0;
    bool wraps = op == Op::ADD ? __builtin_add_overflow(x, y, &v) : __builtin_mul_overflow(x, y, &v);
    int n = wraps ? -1 : combine(c, op, lit, p);
    if ( n < 0 ) {
      out.push_back(p);
      continue;
    }
    c.pool[n].ival = v;
    c.pool[n].lit = double(v);
    lit = n;
  }
  if ( lit >= 0 ) out.push_back(lit);
  if ( acc >= 0 ) out.push_back(acc);
  return out;
}

int emit( Ctx &c, const Expr &e, const std::map<std::string, int> &paths ) {
  if ( e.kind == Expr::ATOM ) {
    const Atom &a = c.atoms[e.atom];
    ENode n;
    n.isAtom = true;
    n.atom = e.atom;
    n.type = a.type;
    n.secret = a.secret;
    if ( a.kind == AtomKind::INT && a.type->is(c.M->getIntType()) ) n.ival = a.ival;
    if ( a.kind == AtomKind::INT ) n.lit = double(a.ival);
    if ( a.kind == AtomKind::FLOAT ) n.lit = a.fval;
    if ( a.secret ) {
      auto it = paths.find(a.key);
      if ( it == paths.end() ) return -1;
      n.need = -it->second;
    }
    c.pool.push_back(n);
    return int(c.pool.size()) - 1;
  }
  Op op = e.kind == Expr::SUM ? Op::ADD : Op::MUL;
  std::vector<int> pub, items;
  for ( auto &k : e.kids ) {
    int n = emit(c, k, paths);
    if ( n < 0 ) return -1;
    (c.pool[n].secret ? items : pub).push_back(n);
  }
  auto pubs = combinePublics(c, op, pub);
  if ( !pubs ) return -1;
  // (an operand of public terms only must come out as one value)
  if ( items.empty() && pubs->size() != 1 ) return -1;
  for ( int p : *pubs ) items.push_back(p);
  while ( items.size() > 1 ) {
    // the two least deep operands first (a public operand has no depth): the deepest secure atom gets the fewest
    // products above it. Two public operands left apart by combinePublics go to secure ones, one at a time.
    size_t i = 0;
    for ( size_t j = 1; j < items.size(); ++j )
      if ( c.pool[items[j]].need < c.pool[items[i]].need ) i = j;
    size_t k = items.size();
    for ( size_t j = 0; j < items.size(); ++j ) {
      if ( j == i || (!c.pool[items[i]].secret && !c.pool[items[j]].secret) ) continue;
      if ( k == items.size() || c.pool[items[j]].need < c.pool[items[k]].need ) k = j;
    }
    if ( k == items.size() ) return -1;
    size_t a = std::min(i, k), b = std::max(i, k);
    int n = combine(c, op, items[a], items[b]);
    if ( n < 0 ) return -1;
    items[a] = n;
    items.erase(items.begin() + long(b));
  }
  return items[0];
}

void emittedCost( Ctx &c, int n, Cost &cost ) {
  if ( c.pool[n].isAtom ) {
    atomMatmuls(c, c.pool[n].atom, nullptr, &cost.mm);
    return;
  }
  int l = c.pool[n].l, r = c.pool[n].r;
  emittedCost(c, l, cost);
  emittedCost(c, r, cost);
  bool sl = c.pool[l].secret, sr = c.pool[r].secret;
  if ( c.pool[n].op == Op::MUL ) {
    if ( sl && sr ) cost.ss++;
    else if ( sl || sr ) cost.sp++;
  } else if ( sl || sr ) {
    cost.sa++;
  }
}

std::string where( const codon::SrcInfo &info ) {
  return info.file + ":" + std::to_string(info.line) + ":" + std::to_string(info.col);
}

std::string describe( Ctx &c, int n ) {
  const ENode &e = c.pool[n];
  if ( e.isAtom ) {
    const Atom &a = c.atoms[e.atom];
    switch ( a.kind ) {
    case AtomKind::VAR: return a.var->getName();
    case AtomKind::INT: return std::to_string(a.ival);
    case AtomKind::FLOAT: return std::to_string(a.fval);
    case AtomKind::FIELD: return "<" + a.key + ">";
    case AtomKind::TRANSPOSE: return "<" + a.key + ">";
    case AtomKind::MATMUL: return "<" + a.key + ">";
    }
  }
  return "(" + describe(c, e.l) + (e.op == Op::ADD ? " + " : " * ") + describe(c, e.r) + ")";
}

Value *materialize( Ctx &c, int a, const codon::SrcInfo &info ) {
  const Atom &at = c.atoms[a];
  switch ( at.kind ) {
  case AtomKind::VAR: return c.M->N<VarValue>(info, at.var);
  case AtomKind::INT: return c.M->N<IntConst>(info, at.ival, at.type);
  case AtomKind::FLOAT: return c.M->N<FloatConst>(info, at.fval, at.type);
  case AtomKind::FIELD: return c.M->N<ExtractInstr>(info, materialize(c, at.inner, info), at.field);
  case AtomKind::TRANSPOSE:
    return c.M->N<CallInstr>(info, c.M->N<VarValue>(info, at.func),
                             std::vector<Value *>{materialize(c, at.inner, info)});
  case AtomKind::MATMUL:
    return c.M->N<CallInstr>(info, c.M->N<VarValue>(info, at.func),
                             std::vector<Value *>{materialize(c, at.inner, info), materialize(c, at.right, info)});
  }
  return nullptr;
}

// Fresh IR for an emitted expression: every value has one parent
Value *build( Ctx &c, int n, const codon::SrcInfo &info ) {
  const ENode e = c.pool[n];
  if ( e.isAtom ) return materialize(c, e.atom, info);
  Value *l = build(c, e.l, info);
  Value *r = build(c, e.r, info);
  std::vector<Value *> args = e.reflected ? std::vector<Value *>{r, l} : std::vector<Value *>{l, r};
  return c.M->N<CallInstr>(info, c.M->N<VarValue>(info, e.func), args);
}

// ---------------------------------------------------------------- planning

struct Plan {
  int root = -1;
  Cost cost;
  std::vector<Check> checks;
};

void atomsOfKind( Ctx &c, int a, OperandKind k, std::set<int> &out ) {
  const Atom &at = c.atoms[a];
  if ( operandKind(at.type) == k ) out.insert(a);
  if ( at.inner >= 0 ) atomsOfKind(c, at.inner, k, out);
  if ( at.right >= 0 ) atomsOfKind(c, at.right, k, out);
}

void exprAtomsOfKind( Ctx &c, const Expr &e, OperandKind k, std::set<int> &out ) {
  if ( e.kind == Expr::ATOM ) {
    atomsOfKind(c, e.atom, k, out);
    return;
  }
  for ( auto &kid : e.kids ) exprAtomsOfKind(c, kid, k, out);
}

void poolMatmuls( Ctx &c, int n, MatmulCounts &counts ) {
  if ( c.pool[n].isAtom ) {
    atomMatmuls(c, c.pool[n].atom, &counts, nullptr);
    return;
  }
  poolMatmuls(c, c.pool[n].l, counts);
  poolMatmuls(c, c.pool[n].r, counts);
}

// A Sharetensor value's fixed-point flag as run-time state: set when the flag of one of the base Sharetensors in
// `mask` (bits index Plan::guard) is set, or always (a product with a float operand)
struct FpSide {
  uint64_t mask = 0;
  bool always = false;
  bool operator<( const FpSide &o ) const { return std::tie(mask, always) < std::tie(o.mask, o.always); }
};

constexpr size_t MAX_GUARD_BASES = 62;

// The Sharetensor variables and fields an atom reads (those under x.T and x @ y included): each a bit of the mask
// (a public float, or an ndarray of floats, makes a product with a Sharetensor fixed point: x @ w with such a w too)
bool atomFpSide( Ctx &c, int a, std::vector<int> &bases, FpSide &side ) {
  const Atom &at = c.atoms[a];
  if ( !at.secret ) {
    side.always = side.always || isFloatPublic(at.type);
    return true;
  }
  if ( at.kind == AtomKind::VAR || at.kind == AtomKind::FIELD ) {
    if ( operandKind(at.type) != OperandKind::SHARETENSOR ) return true;
    auto it = std::find(bases.begin(), bases.end(), a);
    size_t bit = size_t(it - bases.begin());
    if ( it == bases.end() ) {
      if ( bases.size() >= MAX_GUARD_BASES ) return false;
      bases.push_back(a);
    }
    side.mask |= uint64_t(1) << bit;
    return true;
  }
  if ( at.inner >= 0 && !atomFpSide(c, at.inner, bases, side) ) return false;
  return at.right < 0 || atomFpSide(c, at.right, bases, side);
}

// The flag of pool node n (a product is fixed point when a factor is; a sum is what its secure side is), and the
// conditions for its additions to run: two Sharetensors must agree, and a Sharetensor that gets a float must be fixed
// point (Sequre adds a float's fixed-point encoding to an integer Sharetensor without a word)
bool fpSide( Ctx &c, int n, std::vector<int> &bases, std::set<std::pair<FpSide, FpSide>> &conds, FpSide &out ) {
  const ENode &e = c.pool[n];
  out = FpSide();
  if ( e.isAtom ) return atomFpSide(c, e.atom, bases, out);
  FpSide l, r;
  if ( !fpSide(c, e.l, bases, conds, l) || !fpSide(c, e.r, bases, conds, r) ) return false;
  bool sl = c.pool[e.l].secret, sr = c.pool[e.r].secret;
  if ( !sl && !sr ) {
    // (public operands combined: a float when either is)
    out = FpSide{0, l.always || r.always || isFloatPublic(e.type)};
    return true;
  }
  if ( e.op == Op::MUL ) {
    out = FpSide{l.mask | r.mask, l.always || r.always};
    return true;
  }
  bool stl = sl && operandKind(c.pool[e.l].type) == OperandKind::SHARETENSOR;
  bool str = sr && operandKind(c.pool[e.r].type) == OperandKind::SHARETENSOR;
  const FpSide fixed{0, true};
  if ( stl && str ) {
    if ( l.mask != r.mask || l.always != r.always ) conds.insert({l, r});
  } else if ( stl && !sr && r.always && !l.always ) {
    conds.insert({l, fixed});
  } else if ( str && !sl && l.always && !r.always ) {
    conds.insert({r, fixed});
  }
  out = sl || !sr ? l : r;
  return true;
}

// Realizes a run-time check helper((bases...), (ints)...) -> bool; false when it cannot be built
bool makeCheck( Ctx &c, const std::string &helper, const std::vector<int> &bases,
                const std::vector<std::vector<int64_t>> &ints, Check &check ) {
  std::vector<types::Type *> ts, args;
  for ( int b : bases ) ts.push_back(c.atoms[b].type);
  auto *tupleType = c.M->getTupleType(ts);
  check.bases = bases;
  check.basesTuple = realizeMethodQuiet(c.M, tupleType, "__new__", ts);
  if ( !check.basesTuple ) return false;
  args.push_back(tupleType);
  for ( auto &vals : ints ) {
    std::vector<types::Type *> its(vals.size(), c.M->getIntType());
    auto *intsType = c.M->getTupleType(its);
    auto *intsNew = realizeMethodQuiet(c.M, intsType, "__new__", its);
    if ( !intsNew ) return false;
    check.ints.push_back({intsNew, vals});
    args.push_back(intsType);
  }
  check.helper = realizeOptimizationHelperQuiet(c.M, helper, args);
  auto *ret = returnTypeOf(check.helper);
  return ret && ret->is(c.M->getBoolType());
}

// The fixed-point check of a plan, if it adds Sharetensors whose flags may differ: false when it cannot be built
bool fpCheck( Ctx &c, Plan &plan ) {
  std::vector<int> bases;
  std::set<std::pair<FpSide, FpSide>> conds;
  FpSide root;
  if ( !fpSide(c, plan.root, bases, conds, root) ) return false;
  if ( conds.empty() ) return true;
  std::vector<int64_t> vals;
  for ( auto &lr : conds ) {
    vals.push_back(int64_t(lr.first.mask));
    vals.push_back(lr.first.always ? 1 : 0);
    vals.push_back(int64_t(lr.second.mask));
    vals.push_back(lr.second.always ? 1 : 0);
  }
  Check check;
  if ( !makeCheck(c, "fp_agree", bases, {vals}, check) ) return false;
  plan.checks.push_back(check);
  return true;
}

// The secure HE and MP atoms of an expression (Ciphertensor[Ciphertext], MPP, MPA, MPU), each counted once
void exprStorageAtoms( Ctx &c, const Expr &e, std::set<int> &out ) {
  if ( e.kind != Expr::ATOM ) {
    for ( auto &k : e.kids ) exprStorageAtoms(c, k, out);
    return;
  }
  const Atom &at = c.atoms[e.atom];
  auto k = operandKind(at.type);
  if ( at.secret && (k == OperandKind::CIPHER_CT || isMPKind(k)) ) out.insert(e.atom);
}

// The best plan for `written` (definitions folded in), or nothing when it cannot be typed as `type`, costs no less than
// `baseline`, evaluates an `@` product more often than the original (`matmuls`) or puts some secure atom under more
// secure products than the original does
std::optional<Plan> planExpr( Ctx &c, const Expr &written, types::Type *type, const Cost &baseline,
                              const MatmulCounts &matmuls ) {
  // Ciphertensors and MP tensors carry storage the types do not show (encodings, transposition, partitioning, the
  // aggregate), converted for an operation of two stored differently, and their products refresh the right operand in
  // place. A plan that pairs two of them anew can change all of these: such expressions stay as written until a layout
  // planner (BOSTAN, Shechi 8.2) decides storage; one secure tensor of these kinds, with public operands, is planned.
  std::set<int> storage;
  exprStorageAtoms(c, written, storage);
  if ( storage.size() > 1 ) return {};
  Cost wcost;
  std::map<std::string, int> paths;
  writtenCost(c, written, 0, wcost, paths);
  Expr e = written;
  normalize(c, e);
  factor(c, e);
  std::set<int> sts;
  exprAtomsOfKind(c, written, OperandKind::SHARETENSOR, sts);
  c.sharetensorPlan = !sts.empty();
  int root = emit(c, e, paths);
  if ( root < 0 || !c.pool[root].type || !type || !c.pool[root].type->is(type) ) return {};
  if ( c.pool[root].need != NONE && c.pool[root].need > 0 ) return {};
  Cost cost;
  emittedCost(c, root, cost);
  if ( !(cost < baseline) ) return {};
  MatmulCounts planned;
  poolMatmuls(c, root, planned);
  for ( auto &kv : planned ) {
    auto it = matmuls.find(kv.first);
    if ( it == matmuls.end() || kv.second > it->second ) return {};
  }
  Plan plan{root, cost};
  // Sequre refuses to add a fixed-point and an integer Sharetensor (and adds a float to an integer one wrongly), and
  // the flag is run-time state: the plan runs when each addition it makes is of a kind, the expression as written
  // otherwise (no check, no plan)
  if ( !fpCheck(c, plan) ) return {};
  return plan;
}

std::string checksNote( const Plan &plan ) {
  std::string note;
  for ( auto &ch : plan.checks ) note += " [if " + ch.helper->getUnmangledName() + "]";
  return note;
}

void planLocal( Ctx &c, Value *parent, Value *v ) {
  if ( !v || isA<Flow>(v) ) return;
  Op op = Op::ADD;
  Value *l = nullptr, *r = nullptr;
  if ( asOperator(v, op, l, r) ) {
    Expr t;
    if ( parseExpr(c, v, t) ) {
      Cost w;
      std::map<std::string, int> p;
      writtenCost(c, t, 0, w, p);
      MatmulCounts mm;
      exprMatmuls(c, t, mm);
      if ( auto plan = planExpr(c, t, v->getType(), w, mm) ) {
        if ( irOptDebug() )
          std::cerr << "[mhe_cipher_opt] " << c.f->getUnmangledName() << " " << where(v->getSrcInfo()) << ": " << w.str()
                    << " -> " << plan->cost.str() << ": " << describe(c, plan->root) << checksNote(*plan)
                    << std::endl;
        c.rewrites.push_back({parent, v, plan->root, v->getSrcInfo(), plan->checks, {}});
        return;
      }
    }
    planLocal(c, v, l);
    planLocal(c, v, r);
    return;
  }
  for ( auto *u : v->getUsedValues() ) planLocal(c, v, u);
}

struct Def {
  bool pure = false;  // a local variable set to an expression of plain atoms
  Var *lhs = nullptr;
  Expr rhs;
  Cost written;
  MatmulCounts matmuls;
  Cost standalone;    // its own best cost, without folding
};

// Plans a top-level statement whose value is an expression, folding definitions into it where the rules allow.
// Returns false when it leaves the statement to planLocal.
bool planStatement( Ctx &c, const std::vector<SeriesIt> &its, const std::vector<Def> &defs,
                    std::vector<bool> &consumed, size_t i, Value *stmt, Value *root, const Expr &written ) {
  size_t w = i;
  while ( w > 0 && defs[w - 1].pure ) --w;

  std::vector<size_t> cand;
  for ( size_t j = w; j < i; ++j ) {
    if ( consumed[j] ) continue;
    auto *v = defs[j].lhs;
    auto id = v->getId();
    // (a variable nothing reads is never folded, and takes no place among the candidates)
    if ( c.params.count(id) || v->isGlobal() || countOf(c.assigns, id) != 1 || countOf(c.pointers, id) ||
         countOf(c.reads, id) == 0 )
      continue;
    std::map<IrId, int> rr;
    varReads(c, defs[j].rhs, rr, false);
    if ( rr.count(id) ) continue;
    bool stable = true;
    for ( size_t k = j + 1; k < i && stable; ++k )
      if ( rr.count(defs[k].lhs->getId()) ) stable = false;
    if ( stable ) cand.push_back(j);
  }
  // (the masks below have 64 bits: the 64 candidates nearest the statement; a variable a dropped candidate reads
  // keeps a read outside and is not folded)
  if ( cand.size() > 64 ) cand.erase(cand.begin(), cand.end() - 64);

  std::map<IrId, int> stmtReads;
  varReads(c, written, stmtReads, true);
  std::vector<std::map<IrId, int>> defReads(cand.size());
  for ( size_t k = 0; k < cand.size(); ++k ) varReads(c, defs[cand[k]].rhs, defReads[k], true);
  // Every read of a folded variable must be one the folding replaces
  auto readsCovered = [&]( uint64_t mask, size_t k ) {
    auto id = defs[cand[k]].lhs->getId();
    int inside = stmtReads.count(id) ? stmtReads.at(id) : 0;
    for ( size_t m = 0; m < cand.size(); ++m )
      if ( ((mask >> m) & 1) && defReads[m].count(id) ) inside += defReads[m].at(id);
    // (a variable nothing reads is not folded: that would be deleting dead code)
    return inside > 0 && inside == countOf(c.reads, id);
  };
  std::vector<uint64_t> masks;
  if ( cand.size() <= 6 ) {
    for ( uint64_t m = 0; m < (uint64_t(1) << cand.size()); ++m ) {
      bool ok = true;
      for ( size_t k = 0; k < cand.size() && ok; ++k )
        if ( ((m >> k) & 1) && !readsCovered(m, k) ) ok = false;
      if ( ok ) masks.push_back(m);
    }
  } else {
    masks.push_back(0);
    uint64_t m = 0;
    for ( size_t k = 0; k < cand.size() && k < 64; ++k ) m |= uint64_t(1) << k;
    for ( bool changed = true; changed; ) {
      changed = false;
      for ( size_t k = 0; k < cand.size() && k < 64; ++k )
        if ( ((m >> k) & 1) && !readsCovered(m, k) ) {
          m &= ~(uint64_t(1) << k);
          changed = true;
        }
    }
    if ( m ) masks.push_back(m);
  }

  Cost stmtWritten;
  {
    std::map<std::string, int> p;
    writtenCost(c, written, 0, stmtWritten, p);
  }
  MatmulCounts stmtMatmuls;
  exprMatmuls(c, written, stmtMatmuls);
  bool found = false;
  uint64_t bestMask = 0;
  int bestFolds = 0;
  Plan best;
  Cost bestScore = stmtWritten;
  for ( auto m : masks ) {
    Expr combined = written;
    Cost baseline = stmtWritten, standalone;
    MatmulCounts matmuls = stmtMatmuls;
    std::unordered_map<IrId, const Expr *> subst;
    int folds = 0;
    for ( size_t k = 0; k < cand.size(); ++k )
      if ( (m >> k) & 1 ) {
        const Def &d = defs[cand[k]];
        subst[d.lhs->getId()] = &d.rhs;
        baseline = baseline + d.written;
        standalone = standalone + d.standalone;
        for ( auto &kv : d.matmuls ) matmuls[kv.first] += kv.second;
        ++folds;
      }
    if ( folds ) {
      // (substitute in steps, checking the size as it grows)
      bool tooBig = false;
      for ( int round = 0; round < 64 && !tooBig; ++round ) {
        std::unordered_map<IrId, const Expr *> one;
        bool more = false;
        std::map<IrId, int> rr;
        varReads(c, combined, rr, true);
        for ( auto &kv : rr )
          if ( subst.count(kv.first) ) {
            one[kv.first] = subst.at(kv.first);
            more = true;
          }
        if ( !more ) break;
        substituteOnce(c, combined, one);
        tooBig = exprSize(combined) > MAX_FOLDED_SIZE;
      }
      std::map<IrId, int> left;
      varReads(c, combined, left, true);
      bool unfolded = false;
      for ( auto &kv : left ) unfolded = unfolded || subst.count(kv.first);
      if ( tooBig || unfolded ) continue;
    }
    auto plan = planExpr(c, combined, root->getType(), baseline, matmuls);
    if ( !plan ) continue;
    Cost score = plan->cost - standalone;
    if ( score < bestScore || (found && score == bestScore && folds < bestFolds) ) {
      found = true;
      bestMask = m;
      best = *plan;
      bestScore = score;
      bestFolds = folds;
    }
  }
  if ( !found ) return false;

  if ( irOptDebug() ) {
    std::cerr << "[mhe_cipher_opt] " << c.f->getUnmangledName() << " " << where(root->getSrcInfo()) << ": "
              << best.cost.str() << ": " << describe(c, best.root);
    for ( size_t k = 0; k < cand.size(); ++k )
      if ( (bestMask >> k) & 1 ) std::cerr << " [folds " << defs[cand[k]].lhs->getName() << "]";
    std::cerr << checksNote(best) << std::endl;
  }
  std::vector<Value *> folded;
  for ( size_t k = 0; k < cand.size(); ++k )
    if ( (bestMask >> k) & 1 ) {
      consumed[cand[k]] = true;
      c.erasures.push_back(its[cand[k]]);
      folded.push_back(*its[cand[k]]);
    }
  c.rewrites.push_back(
      {root == stmt ? static_cast<Value *>(c.body) : stmt, root, best.root, root->getSrcInfo(), best.checks, folded});
  return true;
}

// ---------------------------------------------------------------- @ chains

// The chain's operands left to right, and for each product (in evaluation order) the first operand of its left
// and of its right side
size_t flattenChain( Value *v, std::vector<Value *> &ops, std::vector<std::pair<size_t, size_t>> &products ) {
  Value *l = nullptr, *r = nullptr;
  if ( asMatmul(v, l, r) ) {
    size_t a = flattenChain(l, ops, products);
    size_t b = flattenChain(r, ops, products);
    products.push_back({a, b});
    return a;
  }
  ops.push_back(v);
  return ops.size() - 1;
}

// The order as written: each product's left side as an index into the list of operands as it shrinks
std::vector<int64_t> writtenOrder( size_t n, const std::vector<std::pair<size_t, size_t>> &products ) {
  std::vector<size_t> starts;
  for ( size_t i = 0; i < n; ++i ) starts.push_back(i);
  std::vector<int64_t> order;
  for ( auto &p : products ) {
    size_t idx = 0;
    while ( idx < starts.size() && starts[idx] != p.first ) ++idx;
    order.push_back(int64_t(idx));
    if ( idx + 1 < starts.size() ) starts.erase(starts.begin() + long(idx) + 1);
  }
  return order;
}

bool chainHelper( Ctx &c, Value *root, const std::vector<Value *> &ops, Chain &chain ) {
  if ( !c.mpcType || !c.mpcVar || ops.size() < 3 ) return false;
  // Sharetensors only: their shape cost model is the only one that can cost a whole order before its products exist
  // (another type's product cost depends on the layouts the products come out in)
  auto *t0 = ops[0]->getType();
  if ( !t0 || operandKind(t0) != OperandKind::SHARETENSOR ) return false;
  for ( auto *op : ops ) {
    // one static type (the helper evaluates a list), and plain operands only (they are evaluated first)
    if ( !op->getType() || !op->getType()->is(t0) ) return false;
    Expr e;
    if ( !parseExpr(c, op, e) ) return false;
  }
  std::vector<types::Type *> ts(ops.size(), t0);
  auto *tupleType = c.M->getTupleType(ts);
  chain.tupleNew = realizeMethodQuiet(c.M, tupleType, "__new__", ts);
  if ( !chain.tupleNew ) return false;
  std::vector<types::Type *> ints(ops.size() - 1, c.M->getIntType());
  auto *ordersType = c.M->getTupleType(ints);
  chain.ordersNew = realizeMethodQuiet(c.M, ordersType, "__new__", ints);
  if ( !chain.ordersNew ) return false;
  chain.helper = realizeOptimizationHelperQuiet(c.M, "matmul_chain", {c.mpcType, tupleType, ordersType});
  return chain.helper && returnTypeOf(chain.helper) && returnTypeOf(chain.helper)->is(root->getType());
}

// Chains anywhere in the function, control flow included (a chain's rewrite changes nothing outside it)
void findChains( Ctx &c, Value *parent, Value *v ) {
  if ( !v ) return;
  Value *l = nullptr, *r = nullptr;
  if ( asMatmul(v, l, r) ) {
    std::vector<Value *> ops;
    std::vector<std::pair<size_t, size_t>> products;
    flattenChain(v, ops, products);
    Chain chain{parent, v, nullptr, nullptr, nullptr};
    if ( ops.size() >= 3 && chainHelper(c, v, ops, chain) ) {
      if ( irOptDebug() )
        std::cerr << "[mhe_cipher_opt] " << c.f->getUnmangledName() << " " << where(v->getSrcInfo()) << ": a chain of "
                  << ops.size() << " operands to matmul_chain" << std::endl;
      c.chains.push_back(chain);
      return;
    }
    findChains(c, v, l);
    findChains(c, v, r);
    return;
  }
  for ( auto *u : v->getUsedValues() ) findChains(c, v, u);
}

Value *buildCheck( Ctx &c, const Check &ch, const codon::SrcInfo &info ) {
  std::vector<Value *> bases;
  for ( int b : ch.bases ) bases.push_back(materialize(c, b, info));
  std::vector<Value *> args = {c.M->N<CallInstr>(info, c.M->N<VarValue>(info, ch.basesTuple), bases)};
  for ( auto &t : ch.ints ) {
    std::vector<Value *> vals;
    for ( auto v : t.second ) vals.push_back(c.M->N<IntConst>(info, v, c.M->getIntType()));
    args.push_back(c.M->N<CallInstr>(info, c.M->N<VarValue>(info, t.first), vals));
  }
  return c.M->N<CallInstr>(info, c.M->N<VarValue>(info, ch.helper), args);
}

// The planned expression's IR; for a plan with run-time checks, `check1 and check2 ? planned : as written`, the
// expression as written running the folded definitions' assignments first, each once, in their order (moved from the
// function's body, which drops them)
Value *buildRewrite( Ctx &c, const Rewrite &rw ) {
  Value *planned = build(c, rw.root, rw.info);
  if ( rw.checks.empty() ) return planned;
  Value *fallback = rw.oldValue;
  if ( !rw.folded.empty() ) {
    auto *series = c.M->N<SeriesFlow>(rw.info);
    for ( auto *a : rw.folded ) series->push_back(a);
    fallback = c.M->N<FlowInstr>(rw.info, series, fallback);
  }
  Value *cond = buildCheck(c, rw.checks[0], rw.info);
  for ( size_t i = 1; i < rw.checks.size(); ++i )
    cond = c.M->N<TernaryInstr>(rw.info, cond, buildCheck(c, rw.checks[i], rw.info),
                                c.M->N<BoolConst>(rw.info, false, c.M->getBoolType()));
  return c.M->N<TernaryInstr>(rw.info, cond, planned, fallback);
}

Value *buildChain( Ctx &c, const Chain &ch ) {
  std::vector<Value *> ops;
  std::vector<std::pair<size_t, size_t>> products;
  flattenChain(ch.root, ops, products);
  auto info = ch.root->getSrcInfo();
  std::vector<Value *> orders;
  for ( auto o : writtenOrder(ops.size(), products) ) orders.push_back(c.M->N<IntConst>(info, o, c.M->getIntType()));
  // The operands move into the tuple: the products that held them are dropped with the chain's root
  auto *tuple = c.M->N<CallInstr>(info, c.M->N<VarValue>(info, ch.tupleNew), ops);
  auto *order = c.M->N<CallInstr>(info, c.M->N<VarValue>(info, ch.ordersNew), orders);
  return c.M->N<CallInstr>(info, c.M->N<VarValue>(info, ch.helper),
                           std::vector<Value *>{c.M->N<VarValue>(info, c.mpcVar), tuple, order});
}

} // namespace

void applyCipherOpt( BodiedFunc *f ) {
  Ctx c;
  c.f = f;
  c.M = f->getModule();
  c.body = cast<SeriesFlow>(f->getBody());
  if ( !c.body || f->isGenerator() || f->isAsync() ) return;
  c.sequre = hasSequreAttr(f);
  if ( f->arg_begin() != f->arg_end() && isMPCType(f->arg_front()->getType()) ) {
    c.mpcVar = f->arg_front();
    c.mpcType = c.mpcVar->getType();
  }
  // Under @sequre a secure operator needs the MPC instance as the first argument: without it the expressiveness pass
  // has no rewrite to check against
  if ( c.sequre && !c.mpcType ) return;
  for ( auto it = f->arg_begin(); it != f->arg_end(); ++it ) c.params.insert((*it)->getId());
  collectVarInfo(c, c.body);
  collectPairs(c, c.body);

  std::vector<SeriesIt> its;
  for ( auto it = c.body->begin(); it != c.body->end(); ++it ) its.push_back(it);
  size_t n = its.size();

  std::vector<Def> defs(n);
  for ( size_t i = 0; i < n; ++i ) {
    auto *ai = cast<AssignInstr>(*its[i]);
    if ( !ai || !ai->getLhs() || !ai->getRhs() || ai->getLhs()->isGlobal() ) continue;
    Expr t;
    if ( !parseExpr(c, ai->getRhs(), t) ) continue;
    Def &d = defs[i];
    d.pure = true;
    d.lhs = ai->getLhs();
    d.rhs = std::move(t);
    std::map<std::string, int> p;
    writtenCost(c, d.rhs, 0, d.written, p);
    exprMatmuls(c, d.rhs, d.matmuls);
    d.standalone = d.written;
    if ( auto plan = planExpr(c, d.rhs, ai->getRhs()->getType(), d.written, d.matmuls) ) d.standalone = plan->cost;
  }

  // From the last statement up: a definition goes to the statement that reads it, which is planned first
  std::vector<bool> consumed(n, false);
  for ( size_t i = n; i-- > 0; ) {
    if ( consumed[i] ) continue;
    Value *stmt = *its[i];
    Value *root = nullptr;
    if ( auto *ai = cast<AssignInstr>(stmt) ) root = ai->getRhs();
    else if ( auto *ri = cast<ReturnInstr>(stmt) ) root = ri->getValue();
    else if ( !isA<Flow>(stmt) ) root = stmt;
    Op op = Op::ADD;
    Value *l = nullptr, *r = nullptr;
    Expr written;
    if ( root && asOperator(root, op, l, r) && parseExpr(c, root, written) &&
         planStatement(c, its, defs, consumed, i, stmt, root, written) )
      continue;
    planLocal(c, c.body, stmt);
  }

  // Commit: nothing above changed the function
  for ( auto &rw : c.rewrites ) {
    IrId old = rw.oldValue->getId();
    Value *nv = buildRewrite(c, rw);
    if ( rw.parent->replaceUsedValue(old, nv) != 1 ) irOptInternalError(f, "a planned expression lost its place");
  }
  for ( auto it : c.erasures ) c.body->erase(it);

  // `@` chains, planned on the function as it is now (a planned expression may hold one) and then committed
  findChains(c, c.body, c.body);
  for ( auto &ch : c.chains ) {
    Value *nv = buildChain(c, ch);
    if ( ch.parent->replaceUsedValue(ch.root->getId(), nv) != 1 )
      irOptInternalError(f, "a chain of products lost its place");
  }
}

} // namespace sequre

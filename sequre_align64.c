/* x86-64 heap-vector alignment workaround for Codon (see the launcher's maybe_set_align_shim() and
 * maybe_link_align_shim(), and scripts/apply_avx512_workaround.sh).
 *
 * Codon emits aligned vector moves (vmovaps of 256-bit ymm and, on AVX-512 CPUs, 512-bit zmm registers) for
 * heap-resident Vec[*,8], but GC_malloc only 16-byte-aligns memory, so those moves fault (#GP/SIGSEGV) on x86-64 CPUs
 * with AVX2 or AVX-512. This shim over-aligns the GC allocations that can hold such a vector so the aligned moves are
 * valid. `sequre build` links it into every binary it builds on x86-64 Linux, ahead of libcodonrt.so; the launcher
 * preloads it (LD_PRELOAD) for `sequre run`. Either way it reaches the real allocators with dlsym(RTLD_NEXT, ...).
 *
 * Two things keep the overhead low:
 *   1. Size threshold. An allocation smaller than the alignment can never hold a 64-byte-aligned vector, so anything
 *      < SEQURE_ALIGN takes the normal fast-path allocator untouched. Most allocations are small objects, so this
 *      avoids forcing them all through GC_memalign (which bypasses the thread-local free lists and over-aligns — the
 *      dominant cost).
 *   2. 64-byte alignment. The widest heap-resident SIMD vector in Sequre is Vec[u64,8] / Vec[float,8] (64 B);
 *      list[Tuple[u64xN,u64xN]] is 128 B in size but only needs 64-byte alignment. Vec[u128,8] appears solely as a
 *      module-level constant (statically aligned, never seq_alloc'd). So 64 is sufficient; override with
 *      -DSEQURE_ALIGN=N if that ever changes.
 *
 * Only the collectable allocators are interposed; the *_uncollectable ones are left alone (routing those through a
 * collectable alloc would let the GC free GC roots). Over-aligned atomic (pointer-free) buffers go through
 * GC_memalign and thus become GC-scanned (correctness-safe; some extra GC work).
 *
 * Interior pointers. For sizes below half a heap block (2048 bytes), GC_memalign allocates a larger object and
 * returns an aligned pointer inside it, not the object's base. GC_realloc and GC_free take such a pointer for a base:
 * GC_free clears the bytes after it, past the end of the object, and puts it on a free list. So a block from the
 * aligned path never reaches them: seq_realloc's shrink path copies it into a fresh small block instead, and seq_free
 * frees the object's base (GC_base). Every block of SEQURE_ALIGN bytes or more comes from the aligned path; smaller
 * ones are real object bases.
 *
 * Build: cc -shared -fPIC -O2 -o sequre_align64.so sequre_align64.c -ldl
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>

/* The bug is specific to x86-64 (wide aligned vector stores; ARM NEON is 16-byte and matches the GC). Only interpose
 * the allocators there: on any other architecture this compiles to a symbol-less .so that interposes nothing, so a
 * macOS/arm64 (or Linux/aarch64) build is completely unaffected even if the shim is somehow built and loaded. The
 * build scripts also gate on x86_64; this is belt-and-suspenders against that gate ever being dropped. */
#if defined(__x86_64__) || defined(_M_X64)

#ifndef SEQURE_ALIGN
#define SEQURE_ALIGN 64 /* Vec[u64,8]/Vec[float,8] = 64B; widest heap vector */
#endif

typedef void *(*alloc_fn)(size_t);
typedef void *(*memalign_fn)(size_t, size_t);
typedef void *(*realloc_fn)(void *, size_t, size_t);
typedef void (*free_fn)(void *);
typedef void *(*base_fn)(void *);

static alloc_fn real_alloc;
static alloc_fn real_alloc_atomic;
static memalign_fn gc_memalign;
static realloc_fn real_realloc;
static free_fn real_free;
static base_fn gc_base;
static int resolved; /* set (release) after every pointer above is stored */

static void *lookup(const char *name) {
  void *f = dlsym(RTLD_NEXT, name);
  if (!f) {
    static const char msg[] = "sequre_align64: the Codon runtime (libcodonrt) does not provide ";
    (void)!write(2, msg, sizeof(msg) - 1);
    (void)!write(2, name, strlen(name));
    (void)!write(2, "\n", 1);
    abort();
  }
  return f;
}

/* Resolved on first use, not at load time: under LD_PRELOAD the shim can load before libcodonrt. Threads that race
 * here store the same values, and the release/acquire pair on `resolved` publishes them. */
static void init(void) {
  if (__atomic_load_n(&resolved, __ATOMIC_ACQUIRE)) {
    return;
  }
  __atomic_store_n(&real_alloc, (alloc_fn)lookup("seq_alloc"), __ATOMIC_RELAXED);
  __atomic_store_n(&real_alloc_atomic, (alloc_fn)lookup("seq_alloc_atomic"), __ATOMIC_RELAXED);
  __atomic_store_n(&gc_memalign, (memalign_fn)lookup("GC_memalign"), __ATOMIC_RELAXED);
  __atomic_store_n(&real_realloc, (realloc_fn)lookup("seq_realloc"), __ATOMIC_RELAXED);
  __atomic_store_n(&real_free, (free_fn)lookup("seq_free"), __ATOMIC_RELAXED);
  __atomic_store_n(&gc_base, (base_fn)lookup("GC_base"), __ATOMIC_RELAXED);
  __atomic_store_n(&resolved, 1, __ATOMIC_RELEASE);
}

/* p came from GC_memalign and is not its object's base. GC_memalign's result is SEQURE_ALIGN-aligned, so a pointer
 * that is not cannot be one (and skips the GC_base lookup). */
static int interior(void *p) { return ((uintptr_t)p % SEQURE_ALIGN) == 0 && gc_base(p) != p; }

void *seq_alloc(size_t n) {
  init();
  if (n < SEQURE_ALIGN) {
    return real_alloc(n); /* too small to hold an aligned vector: fast path */
  }
  return gc_memalign(SEQURE_ALIGN, n);
}

void *seq_alloc_atomic(size_t n) {
  init();
  if (n < SEQURE_ALIGN) {
    return real_alloc_atomic(n); /* stays atomic + fast */
  }
  return gc_memalign(SEQURE_ALIGN, n);
}

/* seq_realloc MUST be interposed too. A growable buffer (e.g. a List backing store) is first allocated here via
 * GC_memalign, then grown through seq_realloc. The stock seq_realloc calls GC_realloc on that over-aligned block,
 * which mishandles it and corrupts the heap (observed as a NULL coroutine resume pointer when building
 * `list(range(N))` for N large enough to trigger a regrow). Codon passes the old size, so we can over-align the grown
 * buffer ourselves: allocate a fresh SEQURE_ALIGN-aligned block and copy. A shrink below SEQURE_ALIGN (Dict and Set
 * shrink their tables) keeps the stock path only for real object bases; a block from the aligned path is copied into
 * a fresh small one. The old block is left to the collector in both cases. */
void *seq_realloc(void *p, size_t newsize, size_t oldsize) {
  init();
  if (newsize < SEQURE_ALIGN) {
    if (!p || !interior(p)) {
      return real_realloc(p, newsize, oldsize);
    }
    void *q = real_alloc(newsize);
    memcpy(q, p, oldsize < newsize ? oldsize : newsize);
    return q;
  }
  void *q = gc_memalign(SEQURE_ALIGN, newsize);
  if (p && oldsize) {
    size_t copy = oldsize < newsize ? oldsize : newsize;
    memcpy(q, p, copy);
  }
  return q;
}

/* An explicit free of a block from the aligned path frees its object (GC_free needs the base). */
void seq_free(void *p) {
  init();
  if (p && interior(p)) {
    p = gc_base(p);
  }
  real_free(p);
}

#endif /* x86-64 */

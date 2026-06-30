/* AVX-512 heap-vector alignment workaround for Codon (see
 * scripts/apply_avx512_workaround.sh and the launcher's maybe_set_align_shim()).
 *
 * Codon emits aligned 512-bit moves (vmovaps %zmm) for heap-resident Vec[*,8],
 * but GC_malloc only 16-byte-aligns memory, so those stores fault (#GP/SIGSEGV)
 * on x86-64 CPUs with AVX-512. This LD_PRELOAD shim over-aligns the GC
 * allocations that can hold such a vector so the aligned moves are valid.
 *
 * Two things keep the overhead low:
 *   1. Size threshold. An allocation smaller than the alignment can never hold a
 *      64-byte-aligned vector, so anything < SEQURE_ALIGN takes the normal
 *      fast-path allocator untouched. Most allocations are small objects, so
 *      this avoids forcing them all through GC_memalign (which bypasses the
 *      thread-local free lists and over-aligns — the dominant cost).
 *   2. 64-byte alignment. The widest heap-resident SIMD vector in Sequre is
 *      Vec[u64,8] / Vec[float,8] (64 B); list[Tuple[u64xN,u64xN]] is 128 B in
 *      size but only needs 64-byte alignment. Vec[u128,8] appears solely as a
 *      module-level constant (statically aligned, never seq_alloc'd). So 64 is
 *      sufficient; override with -DSEQURE_ALIGN=N if that ever changes.
 *
 * Only the collectable allocators are interposed; the *_uncollectable ones are
 * left alone (routing those through a collectable alloc would let the GC free GC
 * roots). Over-aligned atomic (pointer-free) buffers go through GC_memalign and
 * thus become GC-scanned (correctness-safe; some extra GC work). GC_memalign is
 * used (not an interior-pointer trick) so the returned pointer is a real object
 * base that the GC's realloc/size routines handle correctly.
 *
 * Build: cc -shared -fPIC -O2 -o sequre_align64.so sequre_align64.c -ldl
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

/* The bug is specific to x86-64 (wide aligned vector stores; ARM NEON is
 * 16-byte and matches the GC). Only interpose the allocators there: on any other
 * architecture this compiles to a symbol-less .so that interposes nothing, so a
 * macOS/arm64 (or Linux/aarch64) build is completely unaffected even if the
 * shim is somehow built and loaded. The build scripts also gate on x86_64; this
 * is belt-and-suspenders against that gate ever being dropped. */
#if defined(__x86_64__) || defined(_M_X64)

#ifndef SEQURE_ALIGN
#define SEQURE_ALIGN 64 /* Vec[u64,8]/Vec[float,8] = 64B; widest heap vector */
#endif

typedef void *(*alloc_fn)(size_t);
typedef void *(*memalign_fn)(size_t, size_t);
typedef void *(*realloc_fn)(void *, size_t, size_t);

static alloc_fn real_alloc;
static alloc_fn real_alloc_atomic;
static memalign_fn gc_memalign;
static realloc_fn real_realloc;

static void init(void) {
  if (!real_alloc) real_alloc = (alloc_fn)dlsym(RTLD_NEXT, "seq_alloc");
  if (!real_alloc_atomic) real_alloc_atomic = (alloc_fn)dlsym(RTLD_NEXT, "seq_alloc_atomic");
  if (!gc_memalign) gc_memalign = (memalign_fn)dlsym(RTLD_NEXT, "GC_memalign");
  if (!real_realloc) real_realloc = (realloc_fn)dlsym(RTLD_NEXT, "seq_realloc");
}

void *seq_alloc(size_t n) {
  if (n < SEQURE_ALIGN) {
    if (!real_alloc) init();
    return real_alloc(n); /* too small to hold an aligned vector: fast path */
  }
  if (!gc_memalign) init();
  return gc_memalign(SEQURE_ALIGN, n);
}

void *seq_alloc_atomic(size_t n) {
  if (n < SEQURE_ALIGN) {
    if (!real_alloc_atomic) init();
    return real_alloc_atomic(n); /* stays atomic + fast */
  }
  if (!gc_memalign) init();
  return gc_memalign(SEQURE_ALIGN, n);
}

/* seq_realloc MUST be interposed too. A growable buffer (e.g. a List backing
 * store) is first allocated here via GC_memalign, then grown through
 * seq_realloc. The stock seq_realloc calls GC_realloc on that over-aligned
 * block, which mishandles it and corrupts the heap (observed as a NULL
 * coroutine resume pointer when building `list(range(N))` for N large enough to
 * trigger a regrow). Codon passes the old size, so we can over-align the grown
 * buffer ourselves: allocate a fresh SEQURE_ALIGN-aligned block and copy. Only
 * the large path is redirected; small reallocations keep the fast stock path
 * (they can never hold a 64-byte vector and never came from GC_memalign while
 * large). */
void *seq_realloc(void *p, size_t newsize, size_t oldsize) {
  if (newsize < SEQURE_ALIGN) {
    if (!real_realloc) init();
    return real_realloc(p, newsize, oldsize);
  }
  if (!gc_memalign) init();
  void *q = gc_memalign(SEQURE_ALIGN, newsize);
  if (p && oldsize) {
    size_t copy = oldsize < newsize ? oldsize : newsize;
    memcpy(q, p, copy);
  }
  return q;
}

#endif /* x86-64 */

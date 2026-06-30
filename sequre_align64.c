/* AVX-512 heap-vector alignment workaround for Codon (see
 * scripts/apply_avx512_workaround.sh and the launcher's maybe_set_align_shim()).
 *
 * Codon emits aligned 512-bit moves (vmovaps %zmm) for heap-resident Vec[*,8],
 * but GC_malloc only 16-byte-aligns memory, so those stores fault (#GP/SIGSEGV)
 * on x86-64 CPUs with AVX-512. This LD_PRELOAD shim over-aligns every
 * *collectable* allocation via GC_memalign so the aligned moves are valid.
 *
 * Only the collectable allocators are interposed; the *_uncollectable ones are
 * left alone (routing those through a collectable alloc would let the GC free
 * GC roots). Atomic (pointer-free) buffers become GC-scanned as a side effect
 * (correctness-safe; minor extra GC work / false retention).
 *
 * Build: gcc -shared -fPIC -O2 -o sequre_align64.so sequre_align64.c -ldl
 */
#define _GNU_SOURCE
#include <stddef.h>
#include <dlfcn.h>

#ifndef SEQURE_ALIGN
#define SEQURE_ALIGN 128 /* covers Vec[u128,8] (128B) and Vec[u64,8]/Vec[float,8] (64B) */
#endif

typedef void *(*memalign_fn)(size_t, size_t);
static memalign_fn real_memalign;

static void *aligned_alloc_(size_t n) {
  if (!real_memalign) {
    real_memalign = (memalign_fn)dlsym(RTLD_NEXT, "GC_memalign");
  }
  return real_memalign(SEQURE_ALIGN, n);
}

void *seq_alloc(size_t n) { return aligned_alloc_(n); }
void *seq_alloc_atomic(size_t n) { return aligned_alloc_(n); }

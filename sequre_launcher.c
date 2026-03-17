#define _GNU_SOURCE

#include <glob.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *detect_existing_path(const char *const *candidates) {
  for (int i = 0; candidates[i] != NULL; i++) {
    if (access(candidates[i], R_OK) == 0) {
      return candidates[i];
    }
  }
  return NULL;
}

static void ensure_default_ssl_env(void) {
  const char *crypto = getenv("SEQURE_LIBCRYPTO_PATH");
  if (!crypto || !*crypto) {
    static const char *const crypto_candidates[] = {
        "/usr/lib/x86_64-linux-gnu/libcrypto.so.3",
        "/usr/lib/aarch64-linux-gnu/libcrypto.so.3",
        "/usr/lib64/libcrypto.so.3",
        "/usr/lib/libcrypto.so.3",
        "/lib/x86_64-linux-gnu/libcrypto.so.3",
        "/lib/aarch64-linux-gnu/libcrypto.so.3",
        "/lib64/libcrypto.so.3",
        "/lib/libcrypto.so.3",
        NULL,
    };
    const char *detected = detect_existing_path(crypto_candidates);
    setenv("SEQURE_LIBCRYPTO_PATH", detected ? detected : "libcrypto.so", 0);
  }

  const char *ssl = getenv("SEQURE_OPENSSL_PATH");
  if (!ssl || !*ssl) {
    static const char *const ssl_candidates[] = {
        "/usr/lib/x86_64-linux-gnu/libssl.so.3",
        "/usr/lib/aarch64-linux-gnu/libssl.so.3",
        "/usr/lib64/libssl.so.3",
        "/usr/lib/libssl.so.3",
        "/lib/x86_64-linux-gnu/libssl.so.3",
        "/lib/aarch64-linux-gnu/libssl.so.3",
        "/lib64/libssl.so.3",
        "/lib/libssl.so.3",
        NULL,
    };
    const char *detected = detect_existing_path(ssl_candidates);
    setenv("SEQURE_OPENSSL_PATH", detected ? detected : "libssl.so", 0);
  }
}

static void cleanup_socks(void) {
  glob_t g;
  if (glob("sock.*", 0, NULL, &g) == 0) {
    for (size_t i = 0; i < g.gl_pathc; i++) {
      unlink(g.gl_pathv[i]);
    }
  }
  globfree(&g);
}

static char *default_codon_path(void) {
  const char *env = getenv("CODON_BIN");
  if (env && *env) {
    return strdup(env);
  }

  const char *home = getenv("HOME");
  if (!home || !*home) {
    return strdup("codon");
  }

  char *p = malloc(PATH_MAX);
  if (!p) {
    return strdup("codon");
  }

  snprintf(p, PATH_MAX, "%s/.codon/bin/codon", home);
  return p;
}

int main(int argc, char **argv) {
  cleanup_socks();
  ensure_default_ssl_env();

  char *codon = default_codon_path();
  if (!codon) {
    fprintf(stderr, "Failed to determine codon executable path.\n");
    return 1;
  }

  if (!getenv("CODON_DEBUG")) {
    setenv("CODON_DEBUG", "lt", 0);
  }

  const int base = 5; /* codon run --disable-opt=... -plugin sequre */
  char **args = calloc((size_t)argc + (size_t)base + 1, sizeof(char *));
  if (!args) {
    fprintf(stderr, "Out of memory.\n");
    free(codon);
    return 1;
  }

  int k = 0;
  args[k++] = codon;
  args[k++] = "run";
  args[k++] = "--disable-opt=core-pythonic-list-addition-opt";
  args[k++] = "-plugin";
  args[k++] = "sequre";

  for (int i = 1; i < argc; i++) {
    args[k++] = argv[i];
  }
  args[k] = NULL;

  execvp(codon, args);
  perror("execvp failed");
  free(args);
  free(codon);
  return 127;
}

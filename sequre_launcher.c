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

static void print_help(const char *codon) {
  printf("Sequre — Secure computation framework\n");
  printf("\n");
  printf("Usage: sequre [build|run] <file.codon> [codon-flags] [-- program-args]\n");
  printf("\n");
  printf("Modes:\n");
  printf("  run    Compile and execute a Sequre program (default)\n");
  printf("  build  Compile a Sequre program to a binary\n");
  printf("\n");
  printf("Runtime flags (passed after the .codon file):\n");
  printf("  --use-ring         Use ring modulus instead of field modulus\n");
  printf("  --skip-mhe-setup   Skip the MHE key-generation setup phase\n");
  printf("  -h, --help         Show this help message\n");
  printf("\n");
  printf("Execution modes:\n");
  printf("  Use @local decorator for local (single-machine) programs\n");
  printf("  Use mpc() function for distributed (multi-machine) programs\n");
  printf("\n");
  printf("Environment variables:\n");
  printf("  CODON_BIN              Path to the codon executable\n");
  printf("  SEQURE_CP_IPS          Comma-separated party IP addresses\n");
  printf("  SEQURE_CERT_DIR        TLS certificate directory (default: certs)\n");
  printf("  SEQURE_CA_CERT_FILE    CA certificate file (default: ca.pem)\n");
  printf("  SEQURE_USE_TLS         Set to 0 to disable TLS (insecure)\n");
  printf("  SEQURE_OPENSSL_PATH    Path to libssl.so\n");
  printf("  SEQURE_LIBCRYPTO_PATH  Path to libcrypto.so\n");
  printf("\n");
  printf("Examples:\n");
  printf("  sequre run my_protocol.codon                     # @local program\n");
  printf("  sequre run my_protocol.codon --skip-mhe-setup    # @local, SMC-only\n");
  printf("  sequre build my_protocol.codon                   # compile to binary\n");
  printf("  sequre run my_protocol.codon 1                   # distributed, party 1\n");
  printf("\n");
  printf("For Codon compiler flags, run: %s run --help\n", codon);
}

int main(int argc, char **argv) {
  cleanup_socks();
  ensure_default_ssl_env();

  char *codon = default_codon_path();
  if (!codon) {
    fprintf(stderr, "Failed to determine codon executable path.\n");
    return 1;
  }

  /* Handle --help / -h before doing anything else */
  if (argc < 2) {
    print_help(codon);
    free(codon);
    return 0;
  }
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      print_help(codon);
      free(codon);
      return 0;
    }
  }

  const int base = 5; /* codon <mode> --disable-opt=... -plugin sequre */
  char **args = calloc((size_t)argc + (size_t)base + 1, sizeof(char *));
  if (!args) {
    fprintf(stderr, "Out of memory.\n");
    free(codon);
    return 1;
  }

  const char *mode = "run";
  int arg_start = 1;
  if (argc > 1 && (!strcmp(argv[1], "build") || !strcmp(argv[1], "run"))) {
    mode = argv[1];
    arg_start = 2;
  }

  int k = 0;
  args[k++] = codon;
  args[k++] = mode;
  args[k++] = "--disable-opt=core-pythonic-list-addition-opt";
  args[k++] = "-plugin";
  args[k++] = "sequre";

  for (int i = arg_start; i < argc; i++) {
    args[k++] = argv[i];
  }
  args[k] = NULL;

  execvp(codon, args);
  perror("execvp failed");
  free(args);
  free(codon);
  return 127;
}

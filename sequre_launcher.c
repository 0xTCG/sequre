#define _GNU_SOURCE

#include <glob.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

  /* Try to find codon relative to this executable (same bin directory) */
  char self[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (len > 0) {
    self[len] = '\0';
    /* Find last '/' and replace "sequre" with "codon" */
    char *slash = strrchr(self, '/');
    if (slash && (size_t)(slash - self + 6) < sizeof(self)) {
      strcpy(slash + 1, "codon");
      if (access(self, X_OK) == 0) {
        return strdup(self);
      }
    }
  }

  /* Fallback to $HOME/.sequre/bin/codon */
  const char *home = getenv("HOME");
  if (!home || !*home) {
    return strdup("codon");
  }

  char *p = malloc(PATH_MAX);
  if (!p) {
    return strdup("codon");
  }

  snprintf(p, PATH_MAX, "%s/.sequre/bin/codon", home);
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
  printf("  --local            Run the program in local mode (when @main decorator is used)\n");
  printf("  -h, --help         Show this help message\n");
  printf("\n");
  printf("Execution modes:\n");
  printf("  Use @main decorator (recommended) — runs locally with --local,\n");
  printf("  distributed otherwise.\n");
  printf("\n");
  printf("Environment variables:\n");
  printf("  CODON_BIN              Path to the codon executable\n");
  printf("  SEQURE_GMP_PATH        Override auto-detected libgmp path\n");
  printf("  SEQURE_CP_IPS          Comma-separated party IP addresses\n");
  printf("  SEQURE_CERT_DIR        TLS certificate directory (default: certs)\n");
  printf("  SEQURE_CA_CERT_FILE    CA certificate file (default: ca.pem)\n");
  printf("  SEQURE_USE_TLS         Set to 0 to disable TLS (insecure)\n");
  printf("  SEQURE_OPENSSL_PATH    Override auto-detected libssl path\n");
  printf("  SEQURE_LIBCRYPTO_PATH  Override auto-detected libcrypto path\n");
  printf("  CODON_DEBUG            Compilation verbosity (default: t). Set to 0 to silence.\n");
  printf("\n");
  printf("Examples:\n");
  printf("  sequre run my_protocol.codon --local                   # local mode (@main)\n");
  printf("  sequre run my_protocol.codon --local --skip-mhe-setup  # local, MPC-only\n");
  printf("  sequre build my_protocol.codon                         # compile to binary\n");
  printf("  sequre run my_protocol.codon 1                         # distributed, party 1\n");
  printf("  sequre run my_protocol.codon 1 --use-ring              # distributed, party 1, use ring modulus\n");
  printf("\n");
  printf("For Codon compiler flags, run: %s run --help\n", codon);
}

int main(int argc, char **argv) {
  cleanup_socks();
  if (!getenv("CODON_DEBUG")) {
    setenv("CODON_DEBUG", "t", 0); /* show compilation progress by default */
  } else if (strcmp(getenv("CODON_DEBUG"), "0") == 0) {
    unsetenv("CODON_DEBUG"); /* silence compilation output if CODON_DEBUG=0 */
  }

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
  args[k++] = (char *)mode;
  args[k++] = (char *)"--disable-opt=core-pythonic-list-addition-opt";
  args[k++] = (char *)"-plugin";
  args[k++] = (char *)"sequre";

  for (int i = arg_start; i < argc; i++) {
    args[k++] = argv[i];
  }
  args[k] = NULL;

  execvp(codon, args);
  fprintf(stderr, "error: could not find codon at '%s'\n", codon);
  fprintf(stderr, "Set CODON_BIN to the path of your codon executable, e.g.:\n");
  fprintf(stderr, "  export CODON_BIN=/path/to/codon\n");
  free(args);
  free(codon);
  return 127;
}

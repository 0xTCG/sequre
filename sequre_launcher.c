#define _GNU_SOURCE

#include <glob.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <libgen.h>
#endif

/* Removes leftover "sock.<port>" UNIX-domain socket files (see AF_UNIX_PREFIX
 * in stdlib/sequre/constants.codon) from a previous crashed/killed run. */
static void cleanup_socks(void) {
  glob_t g;
  if (glob("sock.*", 0, NULL, &g) == 0) {
    for (size_t i = 0; i < g.gl_pathc; i++) {
      unlink(g.gl_pathv[i]);
    }
    globfree(&g);
  }
}

static char *default_codon_path(void) {
  const char *env = getenv("CODON_BIN");
  if (env && *env) {
    return strdup(env);
  }

  /* Try to find codon relative to this executable (same bin directory).
   * /proc/self/exe is a Linux-only symlink to the running binary's actual
   * path; readlink() resolves it to an absolute path regardless of how the
   * binary was invoked (via PATH, relative path, or symlink) — so the
   * "find last '/'" step below is safe even if sequre was invoked as a bare
   * command name. On macOS this symlink doesn't exist, readlink() returns
   * -1, and we fall through to the $HOME fallback below. */
  char self[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (len > 0) {
    self[len] = '\0';
    /* self is now an absolute path to this executable; replace the
     * "sequre" basename with "codon" to find codon in the same bin dir. */
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

/* Work around a Codon AVX-512 heap-vector alignment bug (aligned vmovaps %zmm to
 * 16-byte-aligned GC memory) by LD_PRELOAD-ing a small shim that over-aligns GC
 * allocations. The shim is resolved next to this executable
 * (<bindir>/../lib/sequre_align64.so). Scoped to the codon child only.
 * Disable with SEQURE_NO_ALIGN_SHIM=1; override the path with SEQURE_ALIGN_SHIM.
 * No-op when the shim is absent (e.g. macOS/arm64, which is unaffected). */
static void maybe_set_align_shim(void) {
  const char *disable = getenv("SEQURE_NO_ALIGN_SHIM");
  if (disable && *disable && strcmp(disable, "0") != 0) {
    return;
  }

  char shim[PATH_MAX];
  const char *override = getenv("SEQURE_ALIGN_SHIM");
  if (override && *override) {
    if (strlen(override) >= sizeof(shim)) {
      return;
    }
    strcpy(shim, override);
  } else {
    char self[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (len <= 0) {
      return; /* e.g. no /proc (macOS); nothing to do */
    }
    self[len] = '\0';
    char *slash = strrchr(self, '/');
    if (!slash) {
      return;
    }
    *slash = '\0'; /* now self == <bindir> */
    if (snprintf(shim, sizeof(shim), "%s/../lib/sequre_align64.so", self) >= (int)sizeof(shim)) {
      return;
    }
  }

  if (access(shim, R_OK) != 0) {
    return; /* shim not installed; run without it */
  }

  const char *existing = getenv("LD_PRELOAD");
  if (existing && *existing) {
    if (strstr(existing, shim) != NULL) {
      return; /* already present */
    }
    size_t n = strlen(shim) + 1 + strlen(existing) + 1;
    char *combined = malloc(n);
    if (!combined) {
      return;
    }
    snprintf(combined, n, "%s:%s", shim, existing);
    setenv("LD_PRELOAD", combined, 1);
    free(combined);
  } else {
    setenv("LD_PRELOAD", shim, 1);
  }
}

static char *default_plugin_path(void) {
  const char *env = getenv("SEQURE_PLUGIN_PATH");
  if (env && *env) {
    return strdup(env);
  }

  /* Resolve the plugin relative to this executable (<bindir>/../lib/codon/
   * plugins/sequre), so the launcher works from any install prefix rather than
   * only ~/.sequre. Mirrors how default_codon_path() finds codon. */
  char self[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (len > 0) {
    self[len] = '\0';
    char *slash = strrchr(self, '/');
    if (slash) {
      *slash = '\0'; /* self == <bindir> */
      char *p = malloc(PATH_MAX);
      if (p) {
        if (snprintf(p, PATH_MAX, "%s/../lib/codon/plugins/sequre", self) < PATH_MAX) {
          return p;
        }
        free(p);
      }
    }
  }

  /* Fallback (e.g. macOS / no /proc): let codon resolve the plugin by name,
   * relative to the codon binary it loads. */
  return strdup("sequre");
}

/* In "run" mode we execvp() straight into codon, which never returns. In
 * "build" mode we need codon to finish and report its exit code so we can
 * conditionally post-process the produced binary (maybe_fix_libomp below) —
 * execvp can't do that since it replaces this process, hence fork+wait. */
static int run_and_wait(char *const args[]) {
  pid_t pid = fork();
  if (pid < 0) {
    return 1;
  }

  if (pid == 0) {
    execvp(args[0], args);
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return 1;
  }

  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }

  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }

  return 1;
}

static void maybe_trace_args(char *const args[]) {
  const char *trace = getenv("SEQURE_TRACE_ARGS");
  if (!trace || !*trace || !strcmp(trace, "0")) {
    return;
  }
  
  fprintf(stderr, "sequre: exec");
  for (int i = 0; args[i] != NULL; i++) {
    fprintf(stderr, " %s", args[i]);
  }
  fputc('\n', stderr);
}

static int has_suffix(const char *s, const char *suffix) {
  size_t ns = strlen(s);
  size_t nf = strlen(suffix);
  if (ns < nf) {
    return 0;
  }
  return strcmp(s + (ns - nf), suffix) == 0;
}

/* Determines the path of the binary that "codon build" is about to produce,
 * by checking for an explicit -o/--output flag or else deriving it from the
 * .codon source filename (codon's own default: basename without ".codon").
 * Needed so maybe_fix_libomp() knows which binary to patch after the build
 * succeeds. */
static char *infer_build_output(int argc, char **argv, int arg_start) {
  const char *output = NULL;
  const char *src = NULL;

  for (int i = arg_start; i < argc; i++) {
    const char *a = argv[i];

    if (!strcmp(a, "-o") || !strcmp(a, "--output")) {
      if (i + 1 < argc) {
        output = argv[i + 1];
      }
      continue;
    }
    if (!strncmp(a, "-o=", 3)) {
      output = a + 3;
      continue;
    }
    if (!strncmp(a, "--output=", 9)) {
      output = a + 9;
      continue;
    }

    if (!src && a[0] != '-' && has_suffix(a, ".codon")) {
      src = a;
    }
  }

  if (output && *output) {
    return strdup(output);
  }

  if (!src) {
    return NULL;
  }

  const char *base = strrchr(src, '/');
  base = base ? base + 1 : src;

  size_t n = strlen(base);
  if (n <= 6 || strcmp(base + n - 6, ".codon") != 0) {
    return NULL;
  }

  char *out = malloc(n - 5);
  if (!out) {
    return NULL;
  }
  memcpy(out, base, n - 6);
  out[n - 6] = '\0';
  return out;
}

static int path_exists(const char *p) { return p && *p && access(p, R_OK) == 0; }

/* macOS-only: codon-built binaries link against libomp.dylib via
 * @loader_path/libomp.dylib or @rpath/libomp.dylib, which only resolves if
 * libomp.dylib happens to sit next to the binary or on the runtime search
 * path. Homebrew installs it under a versioned Cellar path instead, so the
 * binary fails to start with a dyld "Library not loaded" error. The
 * functions below detect that situation (binary_needs_libomp_rewrite, via
 * otool -L) and rewrite the load command to point at a real libomp.dylib
 * (find_libomp + maybe_fix_libomp, via install_name_tool). Linux doesn't
 * need this: libomp.so is found through the standard ELF rpath/ldconfig
 * mechanism and codon links it accordingly. */
#ifdef __APPLE__
static char *find_libomp(const char *codon_bin) {
  const char *env = getenv("SEQURE_LIBOMP_PATH");
  if (path_exists(env)) {
    return strdup(env);
  }

  if (codon_bin && strchr(codon_bin, '/')) {
    char tmp[PATH_MAX];
    if (strlen(codon_bin) < sizeof(tmp)) {
      strcpy(tmp, codon_bin);
      char *dir = dirname(tmp);
      char cand[PATH_MAX];
      snprintf(cand, sizeof(cand), "%s/../lib/codon/libomp.dylib", dir);
      if (path_exists(cand)) {
        return strdup(cand);
      }
    }
  }

  const char *home = getenv("HOME");
  if (home && *home) {
    char cand[PATH_MAX];
    snprintf(cand, sizeof(cand), "%s/.sequre/lib/codon/libomp.dylib", home);
    if (path_exists(cand)) {
      return strdup(cand);
    }
  }

  if (path_exists("/opt/homebrew/opt/libomp/lib/libomp.dylib")) {
    return strdup("/opt/homebrew/opt/libomp/lib/libomp.dylib");
  }
  if (path_exists("/usr/local/opt/libomp/lib/libomp.dylib")) {
    return strdup("/usr/local/opt/libomp/lib/libomp.dylib");
  }

  return NULL;
}

static int binary_needs_libomp_rewrite(const char *binary) {
  char cmd[PATH_MAX * 2];
  snprintf(cmd, sizeof(cmd), "/usr/bin/otool -L '%s' 2>/dev/null", binary);
  FILE *f = popen(cmd, "r");
  if (!f) {
    return 0;
  }

  int needed = 0;
  char line[2048];
  while (fgets(line, sizeof(line), f)) {
    if (strstr(line, "@loader_path/libomp.dylib") || strstr(line, "@rpath/libomp.dylib")) {
      needed = 1;
      break;
    }
  }
  pclose(f);
  return needed;
}

static void maybe_fix_libomp(const char *binary, const char *codon_bin) {
  if (!binary || !*binary || access(binary, X_OK) != 0) {
    return;
  }

  if (!binary_needs_libomp_rewrite(binary)) {
    return;
  }

  char *libomp = find_libomp(codon_bin);
  if (!libomp) {
    fprintf(stderr, "warning: could not locate libomp.dylib for '%s'\n", binary);
    fprintf(stderr, "         set SEQURE_LIBOMP_PATH to fix OpenMP runtime linking automatically\n");
    return;
  }

  char *change1[] = {
      "/usr/bin/install_name_tool", "-change", "@loader_path/libomp.dylib", libomp, (char *)binary, NULL};
  char *change2[] = {
      "/usr/bin/install_name_tool", "-change", "@rpath/libomp.dylib", libomp, (char *)binary, NULL};

  (void)run_and_wait(change1);
  (void)run_and_wait(change2);
  free(libomp);
}
#endif

static void chomp(char *s) {
  size_t n = strlen(s);
  while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r')) {
    s[--n] = '\0';
  }
}

/* Asks the system python3 for its shared-library location via sysconfig.
 * On macOS framework builds (Homebrew, python.org) the library lives under
 * PYTHONFRAMEWORKPREFIX rather than LIBDIR, so both layouts are tried. */
static char *find_libpython_via_sysconfig(void) {
  FILE *f = popen(
      "python3 -c \"import sysconfig as s; "
      "print(s.get_config_var('PYTHONFRAMEWORKPREFIX') or ''); "
      "print(s.get_config_var('LIBDIR') or ''); "
      "print(s.get_config_var('INSTSONAME') or ''); "
      "print(s.get_config_var('LDLIBRARY') or '')\" 2>/dev/null",
      "r");
  if (!f) {
    return NULL;
  }

  char framework_prefix[PATH_MAX] = {0};
  char libdir[PATH_MAX] = {0};
  char instsoname[PATH_MAX] = {0};
  char ldlibrary[PATH_MAX] = {0};
  char *lines[] = {framework_prefix, libdir, instsoname, ldlibrary};
  int ok = 1;
  for (int i = 0; i < 4; i++) {
    if (!fgets(lines[i], PATH_MAX, f)) {
      ok = 0;
      break;
    }
    chomp(lines[i]);
  }
  pclose(f);
  if (!ok) {
    return NULL;
  }

  char cand[PATH_MAX];
  if (*framework_prefix && *instsoname) {
    snprintf(cand, sizeof(cand), "%s/%s", framework_prefix, instsoname);
    if (path_exists(cand)) {
      return strdup(cand);
    }
  }
  if (*libdir && *instsoname) {
    snprintf(cand, sizeof(cand), "%s/%s", libdir, instsoname);
    if (path_exists(cand)) {
      return strdup(cand);
    }
  }
  if (*libdir && *ldlibrary) {
    snprintf(cand, sizeof(cand), "%s/%s", libdir, ldlibrary);
    if (path_exists(cand)) {
      return strdup(cand);
    }
  }
  return NULL;
}

/* Fallback for systems without a usable python3 on PATH: glob common
 * install locations and take the lexicographically last (highest version)
 * match. */
static char *find_libpython_via_glob(void) {
  static const char *patterns[] = {
#ifdef __APPLE__
      "/opt/homebrew/opt/python@3.*/Frameworks/Python.framework/Versions/*/Python",
      "/opt/homebrew/Frameworks/Python.framework/Versions/*/Python",
      "/usr/local/opt/python@3.*/Frameworks/Python.framework/Versions/*/Python",
      "/usr/local/Frameworks/Python.framework/Versions/*/Python",
      "/Library/Frameworks/Python.framework/Versions/*/Python",
      "/opt/homebrew/lib/libpython3*.dylib",
      "/usr/local/lib/libpython3*.dylib",
#else
      "/usr/lib/x86_64-linux-gnu/libpython3*.so",
      "/usr/lib/aarch64-linux-gnu/libpython3*.so",
      "/usr/lib64/libpython3*.so",
      "/usr/lib/libpython3*.so",
      "/usr/local/lib/libpython3*.so",
#endif
      NULL,
  };

  char *best = NULL;
  for (int i = 0; patterns[i]; i++) {
    glob_t g;
    if (glob(patterns[i], 0, NULL, &g) == 0 && g.gl_pathc > 0) {
      const char *match = g.gl_pathv[g.gl_pathc - 1];
      free(best);
      best = strdup(match);
    }
    globfree(&g);
  }
  return best;
}

static void maybe_set_codon_python(void) {
  const char *existing = getenv("CODON_PYTHON");
  if (existing && *existing) {
    return; /* respect user override */
  }

  char *libpython = find_libpython_via_sysconfig();
  if (!libpython) {
    libpython = find_libpython_via_glob();
  }
  if (libpython) {
    setenv("CODON_PYTHON", libpython, 1);
    free(libpython);
  }
}

/* Point SEQURE_GMP_PATH at the libgmp bundled inside the plugin
 * (<plugin>/lib/libgmp.{so,dylib}). The Sequre runtime otherwise only
 * searches a fixed list of *system* locations (see _find_lib in
 * stdlib/sequre/constants.codon), so a clean install with no system libgmp
 * would fail to start even though we ship one. Respects a user-set
 * SEQURE_GMP_PATH and does nothing if the bundled library is absent. */
static void maybe_set_gmp_path(const char *plugin_path) {
  const char *existing = getenv("SEQURE_GMP_PATH");
  if (existing && *existing) {
    return; /* respect user override */
  }
  if (!plugin_path || !*plugin_path) {
    return;
  }

#ifdef __APPLE__
  const char *libname = "libgmp.dylib";
#else
  const char *libname = "libgmp.so";
#endif

  char gmp[PATH_MAX];
  int n = snprintf(gmp, sizeof(gmp), "%s/lib/%s", plugin_path, libname);
  if (n < 0 || (size_t)n >= sizeof(gmp)) {
    return;
  }
  if (path_exists(gmp)) {
    setenv("SEQURE_GMP_PATH", gmp, 1);
  }
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
  printf("  CODON_PYTHON           Path to libpython, for @python interop (auto-detected via python3)\n");
  printf("  SEQURE_GMP_PATH        Override auto-detected libgmp path\n");
  printf("  SEQURE_CP_IPS          Comma-separated party IP addresses\n");
  printf("  SEQURE_CERT_DIR        TLS certificate directory (default: certs)\n");
  printf("  SEQURE_CA_CERT_FILE    CA certificate file (default: ca.pem)\n");
  printf("  SEQURE_USE_TLS         Set to 0 to disable TLS (insecure)\n");
  printf("  SEQURE_OPENSSL_PATH    Override auto-detected libssl path\n");
  printf("  SEQURE_LIBCRYPTO_PATH  Override auto-detected libcrypto path\n");
  printf("  SEQURE_LIBOMP_PATH     Override auto-detected libomp.dylib path (build mode, macOS)\n");
  printf("  SEQURE_PLUGIN_PATH     Override the Sequre plugin directory\n");
  printf("  SEQURE_TRACE_ARGS      Set to 1 to print the exact codon command before execution\n");
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

  maybe_set_codon_python();
  maybe_set_align_shim();

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

  const int base = 4; /* codon <mode> --disable-opt=... --plugin=<path> */
  char **args = calloc((size_t)argc + (size_t)base + 1, sizeof(char *));
  if (!args) {
    fprintf(stderr, "Out of memory.\n");
    free(codon);
    return 1;
  }

  /* default_plugin_path()'s "sequre" fallback (no SEQURE_PLUGIN_PATH/$HOME)
   * is fine as a --plugin=sequre value too: codon's PluginManager::load()
   * (codon/dsl/plugins.cpp) gets the same string regardless of whether it
   * was passed as "-plugin sequre" or "--plugin=sequre" — LLVM's cl::list
   * option parser treats both forms identically — and when the value has
   * no "plugin.toml" next to it, load() falls back to searching
   * "<codon-bin-dir>/../lib/codon/plugins/sequre/plugin.toml". */
  char *plugin_path = default_plugin_path();
  if (!plugin_path) {
    fprintf(stderr, "Out of memory.\n");
    free(args);
    free(codon);
    return 1;
  }

  maybe_set_gmp_path(plugin_path);

  char *plugin_arg = malloc(PATH_MAX + 16);
  if (!plugin_arg) {
    fprintf(stderr, "Out of memory.\n");
    free(plugin_path);
    free(args);
    free(codon);
    return 1;
  }
  snprintf(plugin_arg, PATH_MAX + 16, "--plugin=%s", plugin_path);

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
  args[k++] = plugin_arg;

  /* Codon consumes compiler flags only BEFORE the input file; a -release or
   * -debug placed after <file.codon> is treated as a PROGRAM argument and
   * silently ignored (the program then runs unoptimized -- a 4-14x perf trap
   * on HE kernels). We forward args verbatim, so the user must place such
   * flags right after `sequre run`/`build` (before the input file). Warn
   * loudly if a known compiler flag is detected in program-argument position. */
  static const char *codon_flags[] = {"-release", "--release", "-debug", "--debug", NULL};
  int file_seen = 0;
  for (int i = arg_start; i < argc; i++) {
    if (!strcmp(argv[i], "--")) break;
    if (!file_seen) {
      if (argv[i][0] != '-') file_seen = 1; /* first non-flag token = input file */
      continue;
    }
    for (int f = 0; codon_flags[f]; f++) {
      if (!strcmp(argv[i], codon_flags[f])) {
        fprintf(stderr,
                "Warning: '%s' after the input file is treated as a program "
                "argument and ignored by the compiler. Place it right after "
                "'sequre %s' (before the input file) to enable it.\n",
                argv[i], mode);
        break;
      }
    }
  }

  for (int i = arg_start; i < argc; i++) {
    args[k++] = argv[i];
  }
  args[k] = NULL;

  if (!strcmp(mode, "build")) {
    char *out = infer_build_output(argc, argv, arg_start);
    maybe_trace_args(args);
    int rc = run_and_wait(args);
#ifdef __APPLE__
    if (rc == 0 && out) {
      maybe_fix_libomp(out, codon);
    }
#endif
    free(out);
    free(plugin_arg);
    free(plugin_path);
    free(args);
    free(codon);
    return rc;
  }

  maybe_trace_args(args);
  execvp(codon, args);
  fprintf(stderr, "error: could not find codon at '%s'\n", codon);
  fprintf(stderr, "Set CODON_BIN to the path of your codon executable, e.g.:\n");
  fprintf(stderr, "  export CODON_BIN=/path/to/codon\n");
  free(plugin_arg);
  free(plugin_path);
  free(args);
  free(codon);
  return 127;
}

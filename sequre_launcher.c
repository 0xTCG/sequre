#define _GNU_SOURCE

#include <glob.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <libgen.h>
#include <mach-o/dyld.h>
#endif

/* The x86-64 alignment shim (sequre_align64.c) is preloaded for `sequre run` and linked into the binaries of
 * `sequre build`, on x86-64 Linux only: the aligned-vector fault it works around is x86-64's, and LD_PRELOAD and the
 * ELF NEEDED list are Linux's. */
#if defined(__x86_64__) && defined(__linux__)
#define SEQURE_ALIGN_SHIM_PLATFORM 1
#else
#define SEQURE_ALIGN_SHIM_PLATFORM 0
#endif

/* The directory of this executable as an absolute path with symlinks resolved, or NULL. Linux reads /proc/self/exe;
 * macOS has no /proc and asks dyld (_NSGetExecutablePath, then realpath). Everything the launcher finds by its own
 * location (codon, the plugin, the bundled GMP, the shim) depends on it. */
static char *self_dir(void) {
  char self[PATH_MAX];
#ifdef __APPLE__
  char raw[PATH_MAX];
  uint32_t size = sizeof(raw);
  if (_NSGetExecutablePath(raw, &size) != 0 || !realpath(raw, self)) {
    return NULL;
  }
#else
  ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (len <= 0) {
    return NULL;
  }
  self[len] = '\0';
#endif
  char *slash = strrchr(self, '/');
  if (!slash || slash == self) {
    return NULL;
  }
  *slash = '\0';
  return strdup(self);
}

/* "<dir>/<rest>" in a new buffer, or NULL. */
static char *join_path(const char *dir, const char *rest) {
  size_t n = strlen(dir) + 1 + strlen(rest) + 1;
  char *p = malloc(n);
  if (p) {
    snprintf(p, n, "%s/%s", dir, rest);
  }
  return p;
}

static int path_exists(const char *p) { return p && *p && access(p, R_OK) == 0; }

static char *default_codon_path(const char *bindir) {
  const char *env = getenv("CODON_BIN");
  if (env && *env) {
    return strdup(env);
  }

  /* codon in the launcher's own bin directory */
  if (bindir) {
    char *p = join_path(bindir, "codon");
    if (p && access(p, X_OK) == 0) {
      return p;
    }
    free(p);
  }

  /* Fallback to $HOME/.sequre/bin/codon */
  const char *home = getenv("HOME");
  if (!home || !*home) {
    return strdup("codon");
  }
  return join_path(home, ".sequre/bin/codon");
}

static char *default_plugin_path(const char *bindir) {
  const char *env = getenv("SEQURE_PLUGIN_PATH");
  if (env && *env) {
    return strdup(env);
  }

  /* The plugin relative to this executable (<bindir>/../lib/codon/plugins/sequre), so the launcher works from any
   * install prefix rather than only ~/.sequre. Mirrors how default_codon_path() finds codon. */
  if (bindir) {
    char *p = join_path(bindir, "../lib/codon/plugins/sequre");
    char *toml = p ? join_path(p, "plugin.toml") : NULL;
    int found = path_exists(toml);
    free(toml);
    if (found) {
      return p;
    }
    free(p);
  }

  /* Fallback: let codon resolve the plugin by name, relative to the codon binary it loads
   * (<codon-bin-dir>/../lib/codon/plugins/sequre/plugin.toml). */
  return strdup("sequre");
}

/* The alignment shim to use, or NULL: SEQURE_ALIGN_SHIM, else <bindir>/../lib/sequre_align64.so. NULL when
 * SEQURE_NO_ALIGN_SHIM is set (not "0"), off x86-64 Linux, and when the file is missing (*missing then names it). */
static char *align_shim_path(const char *bindir, char **missing) {
  *missing = NULL;
  if (!SEQURE_ALIGN_SHIM_PLATFORM) {
    return NULL;
  }
  const char *disable = getenv("SEQURE_NO_ALIGN_SHIM");
  if (disable && *disable && strcmp(disable, "0") != 0) {
    return NULL;
  }
  const char *override = getenv("SEQURE_ALIGN_SHIM");
  char *shim = NULL;
  if (override && *override) {
    shim = strdup(override);
  } else if (bindir) {
    shim = join_path(bindir, "../lib/sequre_align64.so");
  }
  if (!shim) {
    return NULL;
  }
  char resolved[PATH_MAX];
  if (!path_exists(shim) || !realpath(shim, resolved)) {
    *missing = shim;
    return NULL;
  }
  free(shim);
  return strdup(resolved);
}

static void warn_missing_shim(const char *missing, const char *mode) {
  if (!missing) {
    return;
  }
  fprintf(stderr,
          "warning: the alignment shim %s is missing. On x86-64, Sequre's MHE code then crashes (SIGSEGV: aligned "
          "vector moves on 16-byte-aligned GC memory)%s. Reinstall Sequre, set SEQURE_ALIGN_SHIM to the shim, or "
          "set SEQURE_NO_ALIGN_SHIM=1 to silence this.\n",
          missing, strcmp(mode, "build") == 0 ? " in the binary built now" : "");
}

/* `sequre run`: LD_PRELOAD the shim into the codon child, which runs the program (JIT). Prepended to any existing
 * LD_PRELOAD, once. */
static void maybe_set_align_shim(const char *shim) {
  if (!shim) {
    return;
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

/* Codon options that take a value, given as the next argument when written without "=" (-o out, -l m, ...). */
static int takes_value(const char *a) {
  static const char *opts[] = {"-o", "--output", "-l", "-D", "-plugin", "--plugin", "-linker-flags", "--linker-flags",
                               "-disable-opt", "--disable-opt", "-numerics", "--numerics", NULL};
  for (int i = 0; opts[i]; i++) {
    if (!strcmp(a, opts[i])) {
      return 1;
    }
  }
  return 0;
}

static int has_prefix(const char *s, const char *prefix) { return strncmp(s, prefix, strlen(prefix)) == 0; }

static int has_suffix(const char *s, const char *suffix) {
  size_t ns = strlen(s);
  size_t nf = strlen(suffix);
  if (ns < nf) {
    return 0;
  }
  return strcmp(s + (ns - nf), suffix) == 0;
}

/* Index of the input file in argv[start..argc), or argc: the first argument ending in ".codon" that is neither an
 * option nor the value of one of the options takes_value() knows, else the first argument that is neither (codon
 * options outside that list, -mcpu native say, take their value from the next argument too). */
static int find_input(int argc, char **argv, int start) {
  int first = argc;
  for (int i = start; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "--")) {
      return first < argc ? first : (i + 1 < argc ? i + 1 : argc);
    }
    if (a[0] == '-') {
      if (takes_value(a)) {
        i++;
      }
      continue;
    }
    if (has_suffix(a, ".codon")) {
      return i;
    }
    if (first == argc) {
      first = i;
    }
  }
  return first;
}

/* -plugin/--plugin options (the "=value" and separate-value forms): codon loads plugins in the order given, and a
 * release build with seq's plugin loaded after Sequre's crashes the compiler, so the launcher passes the user's ahead
 * of Sequre's. Returns 2 for the separate-value form (the option and its value), 1 for "=value", 0 otherwise. */
static int plugin_option(const char *a) {
  if (!strcmp(a, "-plugin") || !strcmp(a, "--plugin")) {
    return 2;
  }
  if (has_prefix(a, "-plugin=") || has_prefix(a, "--plugin=")) {
    return 1;
  }
  return 0;
}

/* `sequre build` on x86-64 Linux: link the shim into the binary, so that it runs without LD_PRELOAD. `-l
 * :sequre_align64.so` puts it in the binary's NEEDED list ahead of libcodonrt.so, which codon links after the -l
 * libraries; the dynamic loader then binds the program's seq_alloc, seq_alloc_atomic, seq_realloc and seq_free to
 * the shim, and the shim reaches libcodonrt's with dlsym(RTLD_NEXT, ...), as under LD_PRELOAD. codon searches, and
 * writes into the binary's RUNPATH, <codon prefix>/lib, which holds the shim in an installed prefix. A shim elsewhere
 * (CODON_BIN or SEQURE_ALIGN_SHIM pointing outside the prefix) needs its directory added with --linker-flags, which
 * codon takes once: when the command line already has one, the binary is built without the shim and a warning says
 * so. codon's own directories come first in the RUNPATH, so when codon's library directory holds a file of the
 * shim's name, the binary loads that one whatever the launcher picked: it is linked by name, and a warning names the
 * file. Returns the number of arguments added to args[k..]. A binary whose shim is missing at run time stops at
 * start with the loader's "sequre_align64.so: cannot open shared object file". */
static int maybe_link_align_shim(char **args, int k, const char *shim, const char *codon, int argc, char **argv,
                                 int arg_start, char **owned) {
  if (!shim) {
    return 0;
  }
  const char *name = strrchr(shim, '/');
  if (!name || name == shim) {
    return 0; /* align_shim_path() returns an absolute path below / */
  }
  name++;
  size_t dirlen = (size_t)(name - shim) - 1;

  /* Does codon's library directory (<codon dir>/../lib) hold a file of the shim's name, and is it the shim? */
  int in_codon_lib = 0;
  char *codon_copy = NULL;
  char *codon_dir = NULL;
  if (strchr(codon, '/')) {
    char resolved[PATH_MAX];
    if (realpath(codon, resolved)) {
      char *slash = strrchr(resolved, '/');
      if (slash) {
        *slash = '\0';
        codon_dir = strdup(resolved);
      }
    }
  }
  if (codon_dir) {
    char *lib = join_path(codon_dir, "../lib");
    char *cand = lib ? join_path(lib, name) : NULL;
    struct stat a, b;
    char resolved[PATH_MAX];
    if (cand && stat(cand, &a) == 0) {
      in_codon_lib = stat(shim, &b) == 0 && a.st_dev == b.st_dev && a.st_ino == b.st_ino;
      codon_copy = strdup(realpath(cand, resolved) ? resolved : cand);
    }
    free(cand);
    free(lib);
    free(codon_dir);
  }

  size_t ln = strlen(name) + 2;
  char *lib_arg = malloc(ln);
  if (!lib_arg) {
    free(codon_copy);
    return 0;
  }
  snprintf(lib_arg, ln, ":%s", name);

  if (codon_copy) {
    if (!in_codon_lib) {
      fprintf(stderr,
              "warning: codon's library directory holds its own %s, %s: the binary will load that one, which comes "
              "first in its library path, not %s (`sequre run` preloads %s).\n",
              name, codon_copy, shim, shim);
    }
    free(codon_copy);
    args[k] = "-l";
    args[k + 1] = lib_arg;
    owned[0] = lib_arg;
    return 2;
  }

  int user_linker_flags = 0;
  for (int i = arg_start; i < argc; i++) {
    if (has_prefix(argv[i], "--linker-flags") || has_prefix(argv[i], "-linker-flags")) {
      user_linker_flags = 1;
    }
  }
  if (user_linker_flags || memchr(shim, ' ', dirlen) || memchr(shim, '\t', dirlen)) {
    fprintf(stderr,
            "warning: the binary is built without the alignment shim %s (it is not in codon's library directory, "
            "and %s). Run the binary with LD_PRELOAD=%s, or add -L<dir> -Wl,-rpath,<dir> for its directory to your "
            "--linker-flags and pass -l :%s.\n",
            shim, user_linker_flags ? "--linker-flags is already given" : "its directory name has a blank", shim,
            name);
    free(lib_arg);
    return 0;
  }
  size_t fn = 2 * dirlen + 64;
  char *flags = malloc(fn);
  if (!flags) {
    free(lib_arg);
    return 0;
  }
  snprintf(flags, fn, "--linker-flags=-L%.*s -Wl,-rpath,%.*s", (int)dirlen, shim, (int)dirlen, shim);
  args[k] = "-l";
  args[k + 1] = lib_arg;
  args[k + 2] = flags;
  owned[0] = lib_arg;
  owned[1] = flags;
  return 3;
}

/* In "run" mode we execvp() straight into codon, which never returns. In "build" mode we need codon to finish and
 * report its exit code so we can conditionally post-process the produced binary (maybe_fix_libomp below) — execvp
 * can't do that since it replaces this process, hence fork+wait. */
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

/* Determines the path of the binary that "codon build" is about to produce: an explicit -o/--output, or else codon's
 * own default, the source path without ".codon" (next to the source, not in the working directory). Needed so
 * maybe_fix_libomp() knows which binary to patch after the build succeeds. */
static char *infer_build_output(int argc, char **argv, int arg_start) {
  const char *output = NULL;
  const char *src = NULL;

  for (int i = arg_start; i < argc; i++) {
    const char *a = argv[i];

    if (!strcmp(a, "-o") || !strcmp(a, "--output")) {
      if (i + 1 < argc) {
        output = argv[i + 1];
      }
      i++;
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
    if (a[0] == '-' && takes_value(a)) {
      i++;
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

  size_t n = strlen(src);
  const char *base = strrchr(src, '/');
  base = base ? base + 1 : src;
  if (strlen(base) <= 6) {
    return NULL; /* ".codon" alone */
  }
  char *out = malloc(n - 5);
  if (!out) {
    return NULL;
  }
  memcpy(out, src, n - 6);
  out[n - 6] = '\0';
  return out;
}

/* macOS-only: codon-built binaries link against libomp.dylib via @loader_path/libomp.dylib or @rpath/libomp.dylib,
 * which only resolves if libomp.dylib happens to sit next to the binary or on the runtime search path. Homebrew
 * installs it under a versioned Cellar path instead, so the binary fails to start with a dyld "Library not loaded"
 * error. The functions below detect that situation (binary_needs_libomp_rewrite, via otool -L) and rewrite the load
 * command to point at a real libomp.dylib (find_libomp + maybe_fix_libomp, via install_name_tool). Linux doesn't need
 * this: libomp.so is found through the standard ELF rpath/ldconfig mechanism and codon links it accordingly. */
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
  /* otool's output through a pipe, without a shell (the path may hold any character) */
  int fds[2];
  if (pipe(fds) != 0) {
    return 0;
  }
  pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return 0;
  }
  if (pid == 0) {
    dup2(fds[1], STDOUT_FILENO);
    close(fds[0]);
    close(fds[1]);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    execl("/usr/bin/otool", "otool", "-L", binary, (char *)NULL);
    _exit(127);
  }
  close(fds[1]);
  FILE *f = fdopen(fds[0], "r");
  int needed = 0;
  if (f) {
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
      if (strstr(line, "@loader_path/libomp.dylib") || strstr(line, "@rpath/libomp.dylib")) {
        needed = 1;
      }
    }
    fclose(f);
  } else {
    close(fds[0]);
  }
  int status = 0;
  waitpid(pid, &status, 0);
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

/* Point SEQURE_GMP_PATH at the libgmp bundled inside the plugin (<plugin>/lib/libgmp.{so,dylib}) for the codon child.
 * The Sequre runtime looks there itself after SEQURE_GMP_PATH (_find_lib and _plugin_lib in
 * stdlib/sequre/constants.codon), through the path of its own source at compile time; with the plugin's stdlib
 * symlinked elsewhere that path leads away from the plugin, and this keeps `sequre run` on the bundled copy.
 * Respects a user-set SEQURE_GMP_PATH and does nothing if the bundled library is absent. */
static void maybe_set_gmp_path(const char *plugin_path) {
  const char *existing = getenv("SEQURE_GMP_PATH");
  if (existing && *existing) {
    return; /* respect user override */
  }
  if (!plugin_path || !*plugin_path) {
    return;
  }

#ifdef __APPLE__
  const char *libname = "lib/libgmp.dylib";
#else
  const char *libname = "lib/libgmp.so";
#endif

  char *gmp = join_path(plugin_path, libname);
  if (path_exists(gmp)) {
    setenv("SEQURE_GMP_PATH", gmp, 1);
  }
  free(gmp);
}

static void print_help(const char *codon) {
  printf("Sequre — Secure computation framework\n");
  printf("\n");
  printf("Usage: sequre [run|build] [codon-flags] <file.codon> [program-args]\n");
  printf("\n");
  printf("Modes:\n");
  printf("  run    Compile and execute a Sequre program (default)\n");
  printf("  build  Compile a Sequre program to a binary; run the binary directly: ./<binary> [program-args]\n");
  printf("\n");
  printf("Codon flags such as -release go before the input file: after it, `sequre run` passes them to the\n");
  printf("program. `sequre build` takes -o <binary> and other codon flags on either side.\n");
  printf("\n");
  printf("Program flags (after the .codon file, or after a built binary):\n");
  printf("  --local            Run all parties on this machine (programs that use the @main decorator)\n");
  printf("  --use-ring         Use ring modulus instead of field modulus\n");
  printf("  --skip-mhe-setup   Skip the MHE key-generation setup phase\n");
  printf("\n");
  printf("Launcher flags (before the input file):\n");
  printf("  -h, --help         Show this help message (after the input file, -h and --help go to the program)\n");
  printf("\n");
  printf("Execution modes:\n");
  printf("  Use @main decorator (recommended) — runs locally with --local,\n");
  printf("  distributed otherwise.\n");
  printf("\n");
  printf("Built binaries run without the launcher. On x86-64 Linux, `sequre build` links Sequre's alignment shim\n");
  printf("into the binary. A binary that uses Python interop needs CODON_PYTHON set (the launcher detects it for\n");
  printf("`sequre run`). Other machines need the same Sequre install at the same path.\n");
  printf("\n");
  printf("Environment variables:\n");
  printf("  CODON_BIN              Path to the codon executable\n");
  printf("  CODON_PYTHON           Path to libpython, for Python interop (auto-detected via python3)\n");
  printf("  SEQURE_GMP_PATH        Override the libgmp path (default: the copy bundled with Sequre)\n");
  printf("  SEQURE_CP_IPS          Comma-separated party IPv4 addresses\n");
  printf("  SEQURE_CERT_DIR        TLS certificate directory (default: certs)\n");
  printf("  SEQURE_CA_CERT_FILE    CA certificate file (default: ca.pem)\n");
  printf("  SEQURE_PARTY_CERT_FILE Party certificate file pattern (default: cp{pid}.pem)\n");
  printf("  SEQURE_PARTY_KEY_FILE  Party private key file pattern (default: cp{pid}-key.pem)\n");
  printf("  SEQURE_PARTY_CERT_NAME Name each party's certificate must carry (default: cp{pid})\n");
  printf("  SEQURE_USE_TLS         Set to 0 to disable TLS (insecure)\n");
  printf("  SEQURE_OPENSSL_PATH    Override auto-detected libssl path\n");
  printf("  SEQURE_LIBCRYPTO_PATH  Override auto-detected libcrypto path\n");
  printf("  SEQURE_LIBOMP_PATH     Override auto-detected libomp.dylib path (build mode, macOS)\n");
  printf("  SEQURE_PLUGIN_PATH     Override the Sequre plugin directory\n");
  printf("  SEQURE_ALIGN_SHIM      Path to the x86-64 alignment shim (default: <prefix>/lib/sequre_align64.so)\n");
  printf("  SEQURE_NO_ALIGN_SHIM   Set to 1 to neither preload nor link the alignment shim (x86-64 Linux)\n");
  printf("  SEQURE_TRACE_ARGS      Set to 1 to print the exact codon command before execution\n");
  printf("  CODON_DEBUG            Compilation verbosity (default: t). Set to 0 to silence.\n");
  printf("\n");
  printf("Examples:\n");
  printf("  sequre run my_protocol.codon --local                   # local mode (@main)\n");
  printf("  sequre run my_protocol.codon --local --skip-mhe-setup  # local, MPC-only\n");
  printf("  sequre run -release my_protocol.codon --local          # optimized build, local mode\n");
  printf("  sequre build -release my_protocol.codon -o my_protocol # compile to a binary\n");
  printf("  ./my_protocol --local                                  # run the binary, local mode\n");
  printf("  sequre run my_protocol.codon 1                         # distributed, party 1\n");
  printf("  sequre run my_protocol.codon 1 --use-ring              # distributed, party 1, use ring modulus\n");
  printf("\n");
  printf("For Codon compiler flags, run: %s run --help\n", codon);
}

int main(int argc, char **argv) {
  /* The launcher deletes no files: a local run removes its own socket files (stdlib/sequre/network/socket.codon). */
  if (!getenv("CODON_DEBUG")) {
    setenv("CODON_DEBUG", "t", 0); /* show compilation progress by default */
  } else if (strcmp(getenv("CODON_DEBUG"), "0") == 0) {
    unsetenv("CODON_DEBUG"); /* silence compilation output if CODON_DEBUG=0 */
  }

  char *bindir = self_dir();
  char *codon = default_codon_path(bindir);
  if (!codon) {
    fprintf(stderr, "Failed to determine codon executable path.\n");
    return 1;
  }

  const char *mode = "run";
  int arg_start = 1;
  if (argc > 1 && (!strcmp(argv[1], "build") || !strcmp(argv[1], "run"))) {
    mode = argv[1];
    arg_start = 2;
  }

  /* --help / -h before the input file, or no input file at all, shows this help. After the input file they belong
   * to the program (e.g. the test runner's own help). */
  int input = find_input(argc, argv, arg_start);
  int want_help = input >= argc;
  for (int i = arg_start; i < input && i < argc; i++) {
    if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      want_help = 1;
    }
  }
  if (want_help) {
    print_help(codon);
    free(codon);
    free(bindir);
    return 0;
  }

  maybe_set_codon_python();

  char *shim_missing = NULL;
  char *shim = align_shim_path(bindir, &shim_missing);
  warn_missing_shim(shim_missing, mode);
  free(shim_missing);
  if (!strcmp(mode, "run")) {
    maybe_set_align_shim(shim);
  }

  /* default_plugin_path()'s "sequre" fallback is fine as a --plugin=sequre value too: codon's PluginManager::load()
   * (codon/dsl/plugins.cpp) gets the same string regardless of whether it was passed as "-plugin sequre" or
   * "--plugin=sequre" — LLVM's cl::list option parser treats both forms identically — and when the value has no
   * "plugin.toml" next to it, load() falls back to searching "<codon-bin-dir>/../lib/codon/plugins/sequre/plugin.toml". */
  char *plugin_path = default_plugin_path(bindir);
  if (!plugin_path) {
    fprintf(stderr, "Out of memory.\n");
    free(codon);
    return 1;
  }
  const char *plugin_env = getenv("SEQURE_PLUGIN_PATH");
  if (plugin_env && *plugin_env) {
    char *toml = join_path(plugin_path, "plugin.toml");
    int found = path_exists(toml);
    free(toml);
    if (!found) {
      fprintf(stderr, "error: SEQURE_PLUGIN_PATH=%s has no plugin.toml: it must be the Sequre plugin directory\n",
              plugin_path);
      return 1;
    }
  }

  maybe_set_gmp_path(plugin_path);

  size_t pn = strlen(plugin_path) + sizeof("--plugin=");
  char *plugin_arg = malloc(pn);
  /* codon <mode> --disable-opt=... --plugin=<path> [-l :<shim> [--linker-flags=...]] <user args> NULL */
  char **args = calloc((size_t)argc + 8, sizeof(char *));
  if (!plugin_arg || !args) {
    fprintf(stderr, "Out of memory.\n");
    return 1;
  }
  snprintf(plugin_arg, pn, "--plugin=%s", plugin_path);

  /* The user's plugin options go ahead of Sequre's: those before the input file, and in build mode (where codon
   * parses the options on both sides of the input file) those after it too. */
  int user_plugins_end = !strcmp(mode, "build") ? argc : input;
  char *moved = calloc((size_t)argc + 1, 1);
  if (!moved) {
    fprintf(stderr, "Out of memory.\n");
    return 1;
  }

  int k = 0;
  args[k++] = codon;
  args[k++] = (char *)mode;
  args[k++] = (char *)"--disable-opt=core-pythonic-list-addition-opt";
  for (int i = arg_start; i < user_plugins_end; i++) {
    int n = plugin_option(argv[i]);
    if (n == 2 && i + 1 >= argc) {
      n = 0; /* a trailing -plugin without its value stays where it is, for codon to refuse */
    }
    for (int j = 0; j < n; j++) {
      args[k++] = argv[i + j];
      moved[i + j] = 1;
    }
    if (n) {
      i += n - 1;
    } else if (argv[i][0] == '-' && takes_value(argv[i])) {
      i++;
    }
  }
  args[k++] = plugin_arg;

  char *owned[2] = {NULL, NULL};
  if (!strcmp(mode, "build")) {
    k += maybe_link_align_shim(args, k, shim, codon, argc, argv, arg_start, owned);
  } else {
    /* Codon consumes compiler flags only BEFORE the input file in run mode; a -release or -debug placed after
     * <file.codon> is a PROGRAM argument and silently ignored (the program then runs unoptimized -- a 4-14x perf trap
     * on HE kernels). We forward args verbatim, so warn loudly if a known compiler flag is in program-argument
     * position. (`codon build` parses flags on both sides of the input file.) */
    static const char *codon_flags[] = {"-release", "--release", "-debug", "--debug", NULL};
    for (int i = input + 1; i < argc; i++) {
      if (!strcmp(argv[i], "--")) {
        break;
      }
      for (int f = 0; codon_flags[f]; f++) {
        if (!strcmp(argv[i], codon_flags[f])) {
          fprintf(stderr,
                  "Warning: '%s' after the input file is treated as a program argument and ignored by the compiler. "
                  "Place it right after 'sequre run' (before the input file) to enable it.\n",
                  argv[i]);
          break;
        }
      }
    }
  }

  for (int i = arg_start; i < argc; i++) {
    if (!moved[i]) {
      args[k++] = argv[i];
    }
  }
  args[k] = NULL;
  free(moved);

  if (!strcmp(mode, "build")) {
    char *out = infer_build_output(argc, argv, arg_start);
    maybe_trace_args(args);
    int rc = run_and_wait(args);
    if (rc == 127) {
      fprintf(stderr, "error: could not run codon at '%s' (set CODON_BIN to the path of your codon executable)\n",
              codon);
    }
#ifdef __APPLE__
    if (rc == 0 && out) {
      maybe_fix_libomp(out, codon);
    }
#endif
    free(out);
    free(owned[0]);
    free(owned[1]);
    free(plugin_arg);
    free(plugin_path);
    free(args);
    free(shim);
    free(codon);
    free(bindir);
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
  free(shim);
  free(codon);
  free(bindir);
  return 127;
}

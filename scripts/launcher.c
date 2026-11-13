// MacEcho.app main executable: log first, then exec the menu bar app from the checkout's .venv.
// A Mach-O launcher (not a shell script) so macOS attributes privacy prompts (Desktop folder,
// microphone) to MacEcho itself. ROOT is baked in at build time: -DROOT="\"/path/to/repo\"".
#include <errno.h>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char path[4096];
    const char *home = getenv("HOME");
    snprintf(path, sizeof path, "%s/Library/Logs/MacEcho.log", home ? home : "/tmp");
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) { dup2(fd, STDOUT_FILENO); dup2(fd, STDERR_FILENO); }

    if (chdir(ROOT) != 0) {
        fprintf(stderr, "MacEcho: cannot open %s: %s\n"
                "If this says 'Operation not permitted', allow MacEcho in System Settings > "
                "Privacy & Security > Files and Folders (Desktop), or move the repo out of ~/Desktop.\n",
                ROOT, strerror(errno));
        return 1;
    }
    setenv("PYTHONPATH", ROOT "/src", 1);
    setenv("PYTHONUNBUFFERED", "1", 1);
    // Spawn python as a child instead of exec'ing it: if this signed launcher replaces itself
    // with python, the menu bar never places the status item (frame stays 0x0) when the app
    // is opened from Finder. Staying alive as the bundle's process keeps MacEcho's identity.
    char *argv[] = {"python", "-m", "macecho.menubar", NULL};
    extern char **environ;
    pid_t child;
    int err = posix_spawn(&child, ROOT "/.venv/bin/python", NULL, NULL, argv, environ);
    if (err != 0) {
        fprintf(stderr, "MacEcho: cannot start %s/.venv/bin/python: %s\n", ROOT, strerror(err));
        return 1;
    }
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

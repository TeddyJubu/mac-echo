// MacEcho.app main executable: log first, then run the app on the checkout's .venv Python,
// embedded in this process. ROOT is baked in at build time: -DROOT="\"/path/to/repo\"".
//
// Python runs here rather than as a spawned or exec'd .venv/bin/python because macOS ties the
// Dock icon, "Keep in Dock", the menu bar status item and privacy prompts (Desktop folder,
// microphone) to the process whose executable is the bundle's. A separate python process is a
// different app as far as the Dock is concerned.
#include <Python.h>
#include <errno.h>
#include <fcntl.h>
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

    PyConfig config;
    PyConfig_InitPythonConfig(&config);  // behaves like the python command: reads the env above
    // Claiming to be the venv's python makes Python find .venv/pyvenv.cfg, and so its packages.
    PyStatus status = PyConfig_SetBytesString(&config, &config.program_name, ROOT "/.venv/bin/python");
    if (!PyStatus_Exception(status))
        status = PyConfig_SetBytesString(&config, &config.run_module, "macecho.menubar");
    if (!PyStatus_Exception(status))
        status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        fprintf(stderr, "MacEcho: cannot start Python from %s/.venv: %s\n", ROOT,
                status.err_msg ? status.err_msg : "unknown error");
        return 1;
    }
    return Py_RunMain();  // python -m macecho.menubar
}

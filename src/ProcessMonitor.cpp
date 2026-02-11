#include "ProcessMonitor.h"

#include <iostream>
#include <sstream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>

static std::string buildCommandLine(int argc, char* argv[]) {
    std::stringstream ss;
    ss << argv[0];
    ss << " --execute";
    for (int i = 1; i < argc; ++i) {
        ss << " " << argv[i];
    }
    return ss.str();
}

int monitorProcess(int argc, char* argv[]) {
    std::string cmdLine = buildCommandLine(argc, argv);

    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    if (!CreateProcess(NULL, const_cast<char *>(cmdLine.c_str()), NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        std::cerr << "CreateProcess failed (" << GetLastError() << ").\n";
        return -1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return -1;
}

#else
#include <unistd.h>
#include <sys/wait.h>

static char** createNewArgv(int argc, char* argv[]) {
    char** newArgv = new char*[argc + 2];
    for (int i = 0; i < argc; ++i) {
        newArgv[i] = argv[i];
    }
    newArgv[argc] = new char[strlen("--execute") + 1];
    strcpy(newArgv[argc], "--execute");
    newArgv[argc + 1] = NULL;
    return newArgv;
}

static void cleanNewArgv(char** newArgv, int argc) {
    delete[] newArgv[argc];
    delete[] newArgv;
}

int monitorProcess(int argc, char* argv[]) {
    pid_t pid = fork();

    if (pid == 0) {
        char** newArgv = createNewArgv(argc, argv);
        execv(argv[0], newArgv);
        std::cerr << "Failed to execv." << std::endl;
        cleanNewArgv(newArgv, argc);
        exit(1);
    } else if (pid > 0) {
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            std::cout << "Child process exited normally. Exiting loop." << std::endl;
            return 0;
        } else if (WIFEXITED(status) && WEXITSTATUS(status) == 1) {
            std::cout << "Child process failed to start. Exiting." << std::endl;
            return -1;
        } else {
            std::cout << "Child process terminated abnormally. Restarting..." << std::endl;
            return -1;
        }
    } else {
        std::cerr << "Failed to fork." << std::endl;
        return 1;
    }
}
#endif

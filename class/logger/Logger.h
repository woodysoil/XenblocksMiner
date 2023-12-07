// Logger.h

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>

class Logger {
public:
    Logger(const std::string& baseFilename, size_t maxFileSize);

    void log(const std::string& message);

private:
    std::string baseFilename_;
    std::ofstream outputFile_;
    size_t maxFileSize_;
    size_t currentFileSize_ = 0;
    int fileIndex_ = 0; // To track the current log file

    std::string getCurrentTimestamp();
    size_t getCurrentFileSize(const std::string& filename);
    std::string getCurrentFilename();
    void switchFile();
};

#endif // LOGGER_H

// Logger.h

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>

class Logger {
public:
    Logger(const std::string& baseFilename, size_t maxFileSize);

    void log(const std::string& message);

    static void logToConsole(const std::string& message);
private:
    std::string baseFilename_;
    std::ofstream outputFile_;
    size_t maxFileSize_;
    size_t currentFileSize_ = 0;
    int fileIndex_ = 0; // To track the current log file
    std::mutex mutex_;
    std::string getCurrentTimestamp();
    size_t getCurrentFileSize(const std::string& filename);
    std::string getCurrentFilename();
    void switchFile();
    static std::mutex consoleMutex_;

};

#endif // LOGGER_H

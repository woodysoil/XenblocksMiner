#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>

// Logger class to handle log messages and rotate log files
class Logger {
public:
    // Constructor initializing with base file name and max file size
    Logger(const std::string& baseFilename, size_t maxFileSize)
        : baseFilename_(baseFilename), maxFileSize_(maxFileSize) {
        currentFileSize_ = getCurrentFileSize(getCurrentFilename());
        outputFile_.open(getCurrentFilename(), std::ios::app); // Open in append mode
    }

    // Function to log a message
    void log(const std::string& message) {
        auto timeStamp = getCurrentTimestamp();
        auto fullMessage = timeStamp + " " + message;
        if (currentFileSize_ + fullMessage.length() >= maxFileSize_) {
            switchFile();
        }
        outputFile_ << fullMessage << std::endl;
        currentFileSize_ += fullMessage.length() + 1; // Include newline character
    }

private:
    std::string baseFilename_;
    std::ofstream outputFile_;
    size_t maxFileSize_;
    size_t currentFileSize_ = 0;
    int fileIndex_ = 0; // To track the current log file

    // Function to get current timestamp
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&time);

        std::ostringstream oss;
        oss << std::put_time(&bt, "%m-%d %H:%M");
        return oss.str();
    }

    // Function to get the size of the current log file
    size_t getCurrentFileSize(const std::string& filename) {
        if (std::filesystem::exists(filename)) {
            return std::filesystem::file_size(filename);
        }
        return 0;
    }

    // Function to get the filename of the current log file
    std::string getCurrentFilename() {
        return baseFilename_ + std::to_string(fileIndex_) + ".txt";
    }

    // Function to switch between log files
    void switchFile() {
        outputFile_.close();
        fileIndex_ = (fileIndex_ + 1) % 2; // Toggle between two files
        outputFile_.open(getCurrentFilename(), std::ios::out | std::ios::trunc); // Open new file and truncate
        currentFileSize_ = 0;
    }
};

int main() {
    // Create a logger with maximum file size of 1MB
    Logger logger("log", 1024 * 1024);

    // Log some messages
    logger.log("This is a log message");
    logger.log("This is another log message");

    return 0;
}

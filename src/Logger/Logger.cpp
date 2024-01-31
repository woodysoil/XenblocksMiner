// Logger.cpp

#include "Logger.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <sstream>

std::mutex Logger::consoleMutex_;

Logger::Logger(const std::string& baseFilename, size_t maxFileSize)
    : baseFilename_(baseFilename), maxFileSize_(maxFileSize) {
    currentFileSize_ = getCurrentFileSize(getCurrentFilename());
    outputFile_.open(getCurrentFilename(), std::ios::app); // Open in append mode
}

void Logger::log(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto timeStamp = getCurrentTimestamp();
    auto fullMessage = timeStamp + " " + message;
    if (currentFileSize_ + fullMessage.length() >= maxFileSize_) {
        switchFile();
    }
    outputFile_ << fullMessage << std::endl;
    currentFileSize_ += fullMessage.length() + 1; // Include newline character
}

std::string Logger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&time);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%m-%d %H:%M");
    return oss.str();
}

size_t Logger::getCurrentFileSize(const std::string& filename) {
    if (std::filesystem::exists(filename)) {
        return std::filesystem::file_size(filename);
    }
    return 0;
}

std::string Logger::getCurrentFilename() {
    return baseFilename_ + std::to_string(fileIndex_) + ".txt";
}

void Logger::switchFile() {
    outputFile_.close();
    fileIndex_ = (fileIndex_ + 1) % 2; // Toggle between two files
    outputFile_.open(getCurrentFilename(), std::ios::out | std::ios::trunc); // Open new file and truncate
    currentFileSize_ = 0;
}

void Logger::logToConsole(const std::string& message) {
    std::lock_guard<std::mutex> lock(consoleMutex_);
    std::cout << message;
    std::cout.flush();
}
// main.cpp

#include "Logger.h"

int main() {
    Logger logger("log", 1024 * 1024);
    logger.log("This is a log message");
    logger.log("This is another log message");
    return 0;
}

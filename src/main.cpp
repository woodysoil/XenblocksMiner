#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <map>
#include <regex>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sstream>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>
#include "EthereumAddressValidator.h"

#include "MiningCommon.h"
#include "CudaDevice.h"
#include "MineUnit.h"
#include "AppConfig.h"
#include "Logger.h"
#include "Argon2idHasher.h"
#include <nlohmann/json.hpp>
#include "HttpClient.h"
#include "PowSubmitter.h"
#include "SHA256Hasher.h"
#include "RandomHexKeyGenerator.h"
#include "MachineIDGetter.h"
#include <crow.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

using namespace std;
namespace po = boost::program_options;

#ifdef _WIN32
BOOL ctrlHandler(DWORD fdwCtrlType) {
    switch (fdwCtrlType) {
    case CTRL_C_EVENT:
        std::cout << "Ctrl-C event\n";
        ExitProcess(0);
    default:
        return FALSE;
    }
}
std::string buildCommandLine(int argc, char* argv[]) {
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
char** createNewArgv(int argc, char* argv[]) {
    char** newArgv = new char*[argc + 2];
    for (int i = 0; i < argc; ++i) {
        newArgv[i] = argv[i];
    }
    newArgv[argc] = new char[strlen("--execute") + 1];
    strcpy(newArgv[argc], "--execute");
    newArgv[argc + 1] = NULL;
    return newArgv;
}

void cleanNewArgv(char** newArgv, int argc) {
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

std::string getDifficulty()
{
    HttpClient httpClient;

    try
    {
        HttpResponse response = httpClient.HttpGet("http://xenblocks.io/difficulty", 10); // 10 seconds timeout
        if (response.GetStatusCode() != 200)
        {
            throw std::runtime_error("Failed to get the difficulty: HTTP status code " + std::to_string(response.GetStatusCode()));
        }

        auto json_response = nlohmann::json::parse(response.GetBody());
        return json_response["difficulty"].get<std::string>();
    }
    catch (const nlohmann::json::parse_error &e)
    {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("Error: " + std::string(e.what()));
    }
}

void updateDifficulty()
{
    try
    {
        std::string difficultyStr = getDifficulty();
        int newDifficulty = std::stoi(difficultyStr);

        std::lock_guard<std::mutex> lock(mtx);
        if (globalDifficulty != newDifficulty)
        {
            globalDifficulty = newDifficulty;
            std::cout << "Updated difficulty to " << globalDifficulty << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        // std::cerr << YELLOW << "Error updating difficulty: " << e.what() << RESET << std::endl;
    }
}

void updateDifficultyPeriodically()
{
    while (running)
    {
        updateDifficulty();
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

nlohmann::json vectorToJson(const std::string &machineId, const std::string &accountAddress, const std::vector<std::pair<int, gpuInfo>> &data)
{
    nlohmann::json j;
    nlohmann::json gpuArray = nlohmann::json::array();

    for (const auto &item : data)
    {
        nlohmann::json jItem;
        std::ostringstream os;

        jItem["index"] = item.first;
        jItem["name"] = item.second.name;
        jItem["memory"] = item.second.memory;

        os << std::fixed << std::setprecision(2) << item.second.usingMemory * 100;
        jItem["usingMemory"] = os.str();
        jItem["temperature"] = item.second.temperature;

        os.str("");
        os.clear();
        os << std::fixed << std::setprecision(2) << item.second.hashrate;
        jItem["hashrate"] = os.str();
        jItem["power"] = item.second.power;
        jItem["hashCount"] = item.second.hashCount;
        gpuArray.push_back(jItem);
    }

    j["machineId"] = machineId;
    j["accountAddress"] = accountAddress;
    j["gpuInfos"] = gpuArray;

    return j;
}

void uploadGpuInfos()
{
    while (running)
    {
        auto now = std::chrono::steady_clock::now();
        std::map<int, std::pair<gpuInfo, std::chrono::steady_clock::time_point>> gpuinfos;
        {
            std::lock_guard<std::mutex> lock(globalGpuInfosMutex);
            gpuinfos = globalGpuInfos;
        }
        std::vector<std::pair<int, gpuInfo>> gpuInfos;
        for (const auto &kv : gpuinfos)
        {
            auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - kv.second.second);
            if (duration.count() <= 2)
            {
                gpuInfos.push_back({kv.first, kv.second.first});
            }
        }
        if (gpuInfos.size() == 0)
        {
            std::this_thread::sleep_for(std::chrono::minutes(5));
            continue;
        }
        std::string infoJson = vectorToJson(machineId, globalUserAddress, gpuInfos).dump(-1);
        // std::cout << infoJson << std::endl;
        std::this_thread::sleep_for(std::chrono::minutes(5));
    }
}

std::string getMachineId()
{
    SHA256Hasher hasher;
    try {
        std::string machineId = MachineIDGetter::getMachineId();
        if(machineId.empty()) {
            throw std::runtime_error("Machine ID is empty");
        }
        return hasher.sha256(machineId).substr(0, 16);
    }
    catch (const std::exception& e) {
        RandomHexKeyGenerator keyGenerator;
        return hasher.sha256(keyGenerator.nextRandomKey()).substr(0, 16);
    }

}

std::string getGpuStatsJson() {
    nlohmann::json result;
    nlohmann::json gpuArray = nlohmann::json::array();
    float totalHashrate = 0.0;

    std::lock_guard<std::mutex> guard(globalGpuInfosMutex);

    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

    for (const auto& gpuInfoPair : globalGpuInfos) {
        const auto& gpuInfo = gpuInfoPair.second.first;
        nlohmann::json gpuJson;
        gpuJson["index"] = gpuInfo.index;
        gpuJson["hashrate"] = gpuInfo.hashrate;
        gpuJson["busId"] = gpuInfo.busId;
        totalHashrate += gpuInfo.hashrate;
        gpuArray.push_back(gpuJson);

    }

    result["totalHashrate"] = totalHashrate;
    result["gpus"] = gpuArray;
    result["uptime"] = uptime;
    result["acceptedBlocks"] = globalNormalBlockCount.load() + globalSuperBlockCount.load();
    result["rejectedBlocks"] = globalFailedBlockCount.load();

    return result.dump();
}

crow::SimpleApp app;
void startServer() {
    app.port(42069).multithreaded().run();
}

void runMiningOnDevice(int deviceIndex,
                       SubmitCallback submitCallback,
                       StatCallback statCallback)
{
    cudaError_t cudaStatus = cudaSetDevice(deviceIndex);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaSetDevice failed for device index: " << deviceIndex << std::endl;
        return;
    }
    auto devices = CudaDevice::getAllDevices();
    auto device = devices[deviceIndex];
    // std::cout << "Starting mining on device #" << deviceIndex << ": "
    //           << device.getName() << std::endl;

    while (running)
    {
        MineUnit unit(deviceIndex, globalDifficulty, submitCallback, statCallback);
        if (unit.runMineLoop() < 0)
        {
            std::cerr << "Mining loop failed on device #" << deviceIndex << std::endl;
            break;
        }
    }
}
std::mutex mtx_submit;
std::condition_variable cv;
std::queue<std::function<void()>> taskQueue;
void interruptSignalHandler(int signum)
{
    running = false;
    cv.notify_all();
    app.stop();
}
void workerThread() {
    while (running) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx_submit);
            cv.wait(lock, []{ return !running || !taskQueue.empty(); });

            if (!running && taskQueue.empty()) {
                break;
            }

            task = std::move(taskQueue.front());
            taskQueue.pop();
        }

        task();
    }
}

nlohmann::json inline getStatData() {
    nlohmann::json result;
    nlohmann::json gpuArray = nlohmann::json::array();
    float totalHashrate = 0.0;

    std::lock_guard<std::mutex> guard(globalGpuInfosMutex);

    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

    for (const auto& gpuInfoPair : globalGpuInfos) {
        const auto& gpuInfo = gpuInfoPair.second.first;
        nlohmann::json gpuJson;
        gpuJson["index"] = gpuInfo.index;
        gpuJson["name"] = gpuInfo.name;
        gpuJson["hashrate"] = gpuInfo.hashrate;
        totalHashrate += gpuInfo.hashrate;
        gpuArray.push_back(gpuJson);

    }

    result["machineId"] = machineId;
    result["minerAddr"] = globalUserAddress;
    result["totalHashrate"] = totalHashrate;
    result["gpus"] = gpuArray;
    result["uptime"] = uptime;
    result["acceptedBlocks"] = globalNormalBlockCount.load() + globalSuperBlockCount.load();
    result["rejectedBlocks"] = globalFailedBlockCount.load();

    return result;
}

void UploadDataPeriodically(int uploadPeriod) {
    HttpClient client;
    std::string url = "https://woodyminer.com/api/stat/upload";
    long timeout = 3000;
    int failureCount = 0; 
    int originalUploadPeriod = uploadPeriod;
    std::this_thread::sleep_for(std::chrono::seconds(10));

    while (running) {
        auto data = getStatData();
        auto response = client.HttpPost(url, data, timeout);
        // std::cout << "Server Response: " << response.GetBody() << std::endl;
        // std::cout << "Status Code: " << response.GetStatusCode() << std::endl;
        if (response.GetStatusCode() == 200) {
            failureCount = 0;
            uploadPeriod = originalUploadPeriod;
        } else {
            failureCount++;
            // std::cout << "Upload failed. Consecutive failures: " << failureCount << std::endl;
        }
        if (failureCount >= 10) {
            uploadPeriod *= 2;
            if(uploadPeriod > 600) {
                uploadPeriod = 600;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(uploadPeriod));
    }
}

int main(int argc, const char *const *argv)
{
    bool executeTask = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--execute") {
            executeTask = true;
            break;
        }
    }
    if(!executeTask){
        // std::cout << "Starting the monitor server..." << std::endl;
        int i = 0;
        while(i++ < 42069){
            if (monitorProcess(argc, const_cast<char**>(argv)) >= 0) {
                break;
            }
        }
        return 0;
    }
    // std::cout << "Executing the miner..." << std::endl;
    try {

        po::options_description desc("XenblocksMiner options");
        desc.add_options()
            ("help,h", "display help information")
            ("totalDevFee", po::value<int>(), "set total developer fee")
            ("ecoDevAddr", po::value<std::string>(), "set ecosystem developer address (will receive half of the total dev fee)")
            ("minerAddr", po::value<std::string>(), "set miner address")
            ("execute", "execute the miner otherwise it will run as a mointor server")
            ("saveConfig", "update configuration file with console inputs");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0; // If help information is requested, print help information and exit the program
        }

        // Preload configuration from local file
        AppConfig appConfig(CONFIG_FILENAME);
        if(!vm.count("minerAddr") || !vm.count("totalDevFee")){
            appConfig.load();
        } else {
            appConfig.tryLoad();
        }
        globalUserAddress = appConfig.getAccountAddress();
        globalEcoDevfeeAddress = appConfig.getEcoDevAddr();
        globalDevfeePermillage = appConfig.getDevfeePermillage();

        // Use parsed configuration information
        if (vm.count("totalDevFee")) {
            int totalDevFee = vm["totalDevFee"].as<int>();
            if (totalDevFee < 0 || totalDevFee > 1000) {
                std::cerr << "The argument (" << totalDevFee << ") for total developer fee (0-1000) is invalid." << std::endl;
                return -1;
            }
            globalDevfeePermillage = totalDevFee;
            std::cout << "Total developer fee set to: " << vm["totalDevFee"].as<int>() << "\n";
        }
        if (vm.count("ecoDevAddr")) {
            std::string ecoDevAddr = vm["ecoDevAddr"].as<std::string>();
            EthereumAddressValidator validator;
            if (!validator.isValid(ecoDevAddr)){
                std::cerr << "The argument (" << ecoDevAddr << ") for ecosystem developer fee address (EIP55) is invalid." << std::endl;
                return -1;
            }
            globalEcoDevfeeAddress = ecoDevAddr;
            std::cout << "Ecosystem developer fee address: " << ecoDevAddr << "\n";
        }
        if (vm.count("minerAddr")) {
            std::string userAddr = vm["minerAddr"].as<std::string>();
            EthereumAddressValidator validator;
            if (!validator.isValid(userAddr)){
                std::cerr << "The argument (" << userAddr << ") for miner address (EIP55) is invalid." << std::endl;
                return -1;
            }
            globalUserAddress = userAddr;
            std::cout << "Miner address: " << userAddr << "\n";
        }
        
        EthereumAddressValidator validator;

        if (!globalEcoDevfeeAddress.empty() && !validator.isValid(globalEcoDevfeeAddress)){
            std::cerr << "The argument (" << globalEcoDevfeeAddress << ") for ecosystem developer fee address (EIP55) is invalid." << std::endl;
            return -1;
        }
        if (!validator.isValid(globalUserAddress)){
            std::cerr << "The argument (" << globalUserAddress << ") for miner address (EIP55) is invalid." << std::endl;
            return -1;
        }
        if (globalDevfeePermillage < 0 || globalDevfeePermillage > 1000) {
            std::cerr << "The argument (" << globalDevfeePermillage << ") for total developer fee (0-1000) is invalid." << std::endl;
            return -1;
        }

        signal(SIGINT, interruptSignalHandler);

        if (vm.count("saveConfig")) {
            appConfig.setAccountAddress(globalUserAddress);
            if(!globalEcoDevfeeAddress.empty()){
                appConfig.setEcoDevAddr(globalEcoDevfeeAddress);
            }
            appConfig.setDevfeePermillage(globalDevfeePermillage);
            appConfig.save();
            std::cout << "Configuration file updated with console inputs." << std::endl;
        }

        std::cout << GREEN << "Logged in as " << globalUserAddress 
        << ". Devfee set at " << globalDevfeePermillage << "/1000." 
        << ((globalDevfeePermillage != 0 && !globalEcoDevfeeAddress.empty()) ? " Ecosystem devfee address: " + globalEcoDevfeeAddress : "") 
        << RESET << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Parameter parsing error: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "Unknown error!\n";
        return -1;
    }

    machineId = getMachineId();
    std::cout << "Machine ID: " << machineId << std::endl;

    globalDifficulty = 42069;
    updateDifficulty();
    std::thread difficultyThread(updateDifficultyPeriodically);
    difficultyThread.detach();

    std::thread uploadThread(uploadGpuInfos);
    uploadThread.detach();

    std::thread submitThread(workerThread);
    submitThread.detach();

    Logger logger("log", 1024 * 1024);
    SubmitCallback submitCallback = [&logger](const std::string &hexsalt, const std::string &key, const std::string &hashed_pure, const size_t attempts, const float hashrate) {

        std::function<void()> task = [&logger, hexsalt, key, hashed_pure, attempts, hashrate]()
                                 {
            int difficulty = 40404;
            {
                std::lock_guard<std::mutex> lock(mtx);
                difficulty = globalDifficulty;
            }
            Argon2idHasher hasher(1, difficulty, 1, hexsalt, HASH_LENGTH);
            std::string hashed_data = hasher.generateHash(key);
            // std::cout << "Generated Hash: " << hashed_data << std::endl;
            // std::cout << "Solution meeting the criteria found, submitting: " << hexsalt <<" " << key << std::endl;
            if(hashed_data.find(hashed_pure) == std::string::npos) {
                // std::cout << "Hashed data does not match" << std::endl;
                return;
            }
            std::ostringstream hashrateStream;
            hashrateStream << std::fixed << std::setprecision(2) << hashrate;
            std::string address = "0x" + hexsalt;
            nlohmann::json payload = {
                {"hash_to_verify", hashed_data},
                {"key", key},
                {"account", address},
                {"attempts", std::to_string(attempts)},
                {"hashes_per_second", hashrateStream.str()},
                {"worker", "1"}
            };
            std::cout << std::endl;
            std::cout << "Payload: " << payload.dump(4) << std::endl;
            logger.log(payload.dump(-1));

            int retries = 0;
            int retries_noResponse = 0;
            std::regex pattern(R"(XUNI\d)");
            while (true) {
                if(retries_noResponse >= 10) {
                    std::cout << RED << "No response from server after " << retries_noResponse << " retries" << RESET << std::endl;
                    logger.log("No response from server: " + payload.dump(-1));
                    return;
                }
                try {
                    // std::cout << "Submitting block " << key << std::endl;
                    HttpClient httpClient;
                    HttpResponse response = httpClient.HttpPost("http://xenblocks.io/verify", payload, 10); // 10 seconds timeout
                    // std::cout << "Server Response: " << response.GetBody() << std::endl;
                    // std::cout << "Status Code: " << response.GetStatusCode() << std::endl;
                    if(response.GetBody() == "") {
                        retries_noResponse++;
                        continue;
                    } else {
                        bool errorButFound = false;
                        if(response.GetBody().find("outside of time window") != std::string::npos){
                            std::cout << "Server Response: " << response.GetBody() << std::endl;
                            logger.log(key + " response: " + response.GetBody());
                            return;
                        }
                        if(response.GetBody().find("already exists") != std::string::npos) {
                            errorButFound = true;
                        } else if(response.GetStatusCode() != 500) {
                            std::cout << "Server Response: " << response.GetBody() << std::endl;
                        }
                        if (response.GetStatusCode() == 200 || errorButFound) {
                            if(hashed_pure.find("XEN11") != std::string::npos){
                                size_t capitalCount = std::count_if(hashed_pure.begin(), hashed_pure.end(), [](unsigned char c) { return std::isupper(c); });
                                if (capitalCount >= 50) {
                                    std::cout << GREEN << "Superblock found!" << RESET << std::endl;
                                    globalSuperBlockCount++;
                                } else {
                                    std::cout << GREEN << "Normalblock found!" << RESET << std::endl;
                                    globalNormalBlockCount++;
                                }
                                PowSubmitter::submitPow(address, key, hashed_data);
                                break;
                            } else if (std::regex_search(hashed_pure, pattern)){
                                std::cout << GREEN << "Xuni found!" << RESET << std::endl;
                                globalXuniBlockCount++;
                                break;
                            }
                        }

                        if (response.GetStatusCode() != 500) {
                            logger.log(key + " trying..." + std::to_string(retries + 1) + " response: " + response.GetBody());
                        } else {
                            logger.log(key + " response: status 500");
                        }
                    }

                } catch (const std::exception& e) {
                    // std::cerr << YELLOW <<"An error occurred: " << e.what() << RESET << std::endl;
                } 
                retries++;
                // std::cout << YELLOW << "Retrying... (" << retries << "/" << MAX_SUBMIT_RETRIES << ")" << RESET << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                if (retries >= MAX_SUBMIT_RETRIES) {
                    if(hashed_pure.find("XEN11") != std::string::npos){
                        globalFailedBlockCount++;
                    }
                    std::cout << RED << "Failed to submit block after " << retries << " retries" << RESET << std::endl;
                    logger.log("Failed to submit block: " + payload.dump(-1));
                    return;
                } 
            }
        };

        {
            std::lock_guard<std::mutex> lock(mtx_submit);
            taskQueue.push(std::move(task));
        }
        cv.notify_one();
    };

    StatCallback statCallback = [](const gpuInfo gpuinfo)
    {
        {
            std::lock_guard<std::mutex> lock(globalGpuInfosMutex);
            globalGpuInfos[gpuinfo.index] = {gpuinfo, std::chrono::steady_clock::now()};
        }
        int difficulty = 40404;
        {
            std::lock_guard<std::mutex> lock(mtx);
            difficulty = globalDifficulty;
        }
        size_t totalHashCount = 0;
        float totalHashrate = 0.0;

        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(globalGpuInfosMutex);
            int gpuCount = 0;
            for (const auto &kv : globalGpuInfos)
            {
                auto duration = std::chrono::duration_cast<std::chrono::minutes>(now - kv.second.second);
                if (duration.count() > 2)
                {
                    continue;
                }
                gpuCount++;
                const gpuInfo &info = kv.second.first;
                totalHashCount += info.hashCount;
                totalHashrate += info.hashrate;
            }

            std::ostringstream stream;
            auto elapsed_time = chrono::system_clock::now() - start_time;
	        auto hours = chrono::duration_cast<chrono::hours>(elapsed_time).count();
            auto minutes = chrono::duration_cast<chrono::minutes>(elapsed_time).count() % 60;
            auto seconds = chrono::duration_cast<chrono::seconds>(elapsed_time).count() % 60;
            stream << "\033[2K\r"
                   << "Mining: " << globalHashCount << " Hashes [";
            if (hours > 0) {
                stream << hours << ":";
            }
            stream  << std::setw(2) << std::setfill('0') << minutes << ":";
            stream << std::setw(2) << std::setfill('0') << seconds << ", ";
            stream << gpuCount << " GPUs, ";
            if(globalSuperBlockCount > 0) {
                stream << RED  << " super:" << globalSuperBlockCount<< RESET << ", " ;
            }
            if(globalNormalBlockCount > 0) {
                stream << GREEN << "normal:"  << globalNormalBlockCount << RESET << ", " ;
            }
            if(globalXuniBlockCount > 0) {
                stream << YELLOW << "xuni:"  << globalXuniBlockCount << RESET << ", " ;
            }
            stream << std::fixed << std::setprecision(2) << totalHashrate << " Hashes/s, "
                   << "Difficulty=" << difficulty << "]";
            std::string logMessage = stream.str();
            Logger::logToConsole(logMessage);
        }
        // std::cout << "GPU #" << gpuinfo.index << ": " << gpuinfo.name << std::endl;
        // std::cout << "Memory: " << gpuinfo.memory << "GB" << std::endl;
        // std::cout << "Using Memory: " << gpuinfo.usingMemory * 100 << "%" << std::endl;
        // std::cout << "Temperature: " << gpuinfo.temperature << "C" << std::endl;
        // std::cout << "Hashrate: " << gpuinfo.hashrate << "H/s" << std::endl;
        // std::cout << "Power: " << gpuinfo.power << "W" << std::endl;
        // std::cout << "Hash Count: " << gpuinfo.hashCount << std::endl;
    };
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return -1;
    }

    auto devices = CudaDevice::getAllDevices();

    std::size_t i = 0;
    for (auto &device : devices)
    {
        std::cout << "Device #" << i << ": "
                  << device.getName() << std::endl;
        i++;
    }
    start_time = std::chrono::system_clock::now();
    for (std::size_t i = 0; i < devices.size(); ++i)
    {
        std::thread t(runMiningOnDevice, i, submitCallback, statCallback);
        t.detach();
    }

    app.loglevel(crow::LogLevel::Warning);
    app.signal_clear();
    CROW_ROUTE(app, "/stats")
    ([](){
        auto stats = getGpuStatsJson();
        return stats;
    });
    std::thread serverThread(startServer);
    serverThread.detach();
    std::thread uploadStatThread(UploadDataPeriodically, 60);
    uploadStatThread.detach();
    while (running)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << std::endl;
    return 0;
}

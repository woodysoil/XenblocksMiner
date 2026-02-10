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
#include <nvml.h>
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
#include "MiningCoordinator.h"
#include "PlatformManager.h"
#include <crow.h>
#include <set>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#endif

using namespace std;
namespace po = boost::program_options;
std::string globalCustomName = "";
bool globalPlatformMode = false;
std::string globalMqttBroker = "";
std::string globalWorkerId = "";
std::unique_ptr<PlatformManager> globalPlatformManager;
std::string globalTestBlockPattern = "";

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
        HttpResponse response = httpClient.HttpGet(globalRpcLink+"/difficulty", 5000);
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
        std::this_thread::sleep_for(std::chrono::seconds(10));// Consider a mechanism for proactive notification
        // If this query period increases, we may lose some of the network's computing power when the difficulty changes frequently.
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

std::string getMachineId(string userInputDeviceInfo)
{
    SHA256Hasher hasher;
    try {
        std::string machineId = MachineIDGetter::getMachineId();
        if(machineId.empty()) {
            throw std::runtime_error("Machine ID is empty");
        }
        return hasher.sha256(machineId + userInputDeviceInfo).substr(0, 16);
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
    if (globalPlatformManager) {
        globalPlatformManager->stop();
    }
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
    size_t totalHashCount = 0;

    std::lock_guard<std::mutex> guard(globalGpuInfosMutex);

    auto now = std::chrono::system_clock::now();
    auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    nvmlReturn_t nvmlResult;
    nvmlDevice_t nvmlDevice;
    nvmlUtilization_t nvmlUtilization;
    nvmlMemory_t nvmlMemory;
    unsigned int totalPower;
    nvmlResult = nvmlInit();
    for (const auto& gpuInfoPair : globalGpuInfos) {
        const auto& gpuInfo = gpuInfoPair.second.first;
        nlohmann::json gpuJson;
        gpuJson["index"] = gpuInfo.index;
        gpuJson["name"] = gpuInfo.name;
        std::ostringstream stream_hashRate;
        stream_hashRate << std::fixed << std::setprecision(2) << gpuInfo.hashrate;
        gpuJson["hashrate"] = stream_hashRate.str();
        gpuJson["memory"] = gpuInfo.memory;
        unsigned int power = -1;
        if(nvmlResult == NVML_SUCCESS) {
            nvmlReturn_t nvmlResult_ = nvmlDeviceGetHandleByIndex(gpuInfo.index, &nvmlDevice);
            if (nvmlResult_ == NVML_SUCCESS) {
                nvmlResult_ = nvmlDeviceGetPowerUsage(nvmlDevice, &power);
                nvmlResult_ = nvmlDeviceGetUtilizationRates(nvmlDevice, &nvmlUtilization);
            } 
        } 
        gpuJson["power"] = power;
        totalPower += power == -1 ? 0 : power;
        gpuJson["utiliz"] = nvmlUtilization.gpu;
        std::ostringstream stream_usingMemory;
        stream_usingMemory << std::fixed << std::setprecision(1) << gpuInfo.usingMemory * 100;
        gpuJson["usingMemory"] = stream_usingMemory.str();
        gpuJson["hashCount"] = gpuInfo.hashCount;
        totalHashrate += gpuInfo.hashrate;
        totalHashCount += gpuInfo.hashCount;
        gpuArray.push_back(gpuJson);

    }
    if(nvmlResult == NVML_SUCCESS) {
        nvmlShutdown();
    }

    result["machineId"] = machineId;
    result["minerAddr"] = globalUserAddress;
    std::ostringstream stream_totalHashrate;
    stream_totalHashrate << std::fixed << std::setprecision(2) << totalHashrate;
    result["totalHashrate"] = stream_totalHashrate.str();
    result["totalHashCount"] = totalHashCount;
    result["totalPower"] = totalPower;
    int difficulty = 40404;
    {
        std::lock_guard<std::mutex> lock(mtx);
        difficulty = globalDifficulty;
    }
    result["difficulty"] = difficulty;
    result["gpus"] = gpuArray;
    result["uptime"] = uptime;
    result["acceptedBlocks"] = globalNormalBlockCount.load() + globalSuperBlockCount.load();
    result["normalBlocks"] = globalNormalBlockCount.load();
    result["superBlocks"] = globalSuperBlockCount.load();
    result["rejectedBlocks"] = globalFailedBlockCount.load();
    result["version"] = "2.0.0";
    if (!globalCustomName.empty()) {
        result["customName"] = globalCustomName;
    }
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
        if (response.GetStatusCode() == 201) {
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

std::set<int> parseDeviceList(const std::string& deviceListText, int deviceCount) {
    std::set<int> devices;
    std::stringstream ss(deviceListText);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            int deviceId = std::stoi(item);
            if (deviceId < 0 || deviceId >= deviceCount) {
                continue;
            }
            devices.insert(deviceId);
        } catch (const std::invalid_argument& e) {
            continue;
        } catch (const std::out_of_range& e) {
            continue;
        }
    }
    if (devices.empty()) {
        for (int i = 0; i < deviceCount; ++i) {
            devices.insert(i);
        }
    }
    return devices;
}

int main(int argc, const char *const *argv)
{
    bool executeTask = false;
    bool donotupload = false;
    static bool isTestFixedDiff = false;
    std::string deviceList = "";

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
            ("donotupload", "do not upload the data to the server")
            ("device", po::value<std::string>(), "device index list[--device=1,2,7] to run the miner on")
            ("saveConfig", "update configuration file with console inputs")
            ("testFixedDiff", po::value<int>(), "run in test mode with a fixed difficulty")
            ("rpcLink", po::value<std::string>(), "set rpc link")
            ("customName", po::value<std::string>(), "set custom name")
            ("platform-mode", "enable hashpower marketplace platform mode")
            ("mqtt-broker", po::value<std::string>(), "MQTT broker URI for platform mode (e.g. tcp://broker:1883)")
            ("worker-id", po::value<std::string>(), "override worker ID for platform registration")
            ("testBlockPattern", po::value<std::string>(), "override block detection pattern for testing (default: XEN11)")
            ("batchSize", po::value<int>(), "limit GPU batch size (reduces VRAM usage)");
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0; // If help information is requested, print help information and exit the program
        }

        if(vm.count("testFixedDiff")){
            isTestFixedDiff = true;
            globalDifficulty = vm["testFixedDiff"].as<int>();
        }

        if(vm.count("testBlockPattern")){
            globalTestBlockPattern = vm["testBlockPattern"].as<std::string>();
            std::cout << "Test block pattern override: " << globalTestBlockPattern << std::endl;
        }

        if(vm.count("batchSize")){
            globalMaxBatchSize = vm["batchSize"].as<int>();
            std::cout << "Max batch size override: " << globalMaxBatchSize << std::endl;
        }

        // Preload configuration from local file
        AppConfig appConfig(CONFIG_FILENAME);
        if (!isTestFixedDiff) {
            if(!vm.count("minerAddr") || !vm.count("totalDevFee")){
                appConfig.load();
            } else {
                appConfig.tryLoad();
            }
            globalUserAddress = appConfig.getAccountAddress();
            globalEcoDevfeeAddress = appConfig.getEcoDevAddr();
            globalDevfeePermillage = appConfig.getDevfeePermillage();
        } else {
            globalUserAddress = "0x0000000000000000000000000000000000000000";
            globalEcoDevfeeAddress = "0x0000000000000000000000000000000000000000";
            globalDevfeePermillage = 0;
        }

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

        if (vm.count("donotupload")) {
            donotupload = true;
        }

        if(vm.count("device")){
            deviceList = vm["device"].as<std::string>();
        }

        if (vm.count("rpcLink")) {
            globalRpcLink = vm["rpcLink"].as<std::string>();
        }
        
        if (vm.count("customName")) {
            globalCustomName = vm["customName"].as<std::string>();
        }

        if (vm.count("platform-mode")) {
            globalPlatformMode = true;
        }
        if (vm.count("mqtt-broker")) {
            globalMqttBroker = vm["mqtt-broker"].as<std::string>();
        }
        if (vm.count("worker-id")) {
            globalWorkerId = vm["worker-id"].as<std::string>();
        }

        if (globalPlatformMode && globalMqttBroker.empty()) {
            std::cerr << "Platform mode requires --mqtt-broker to be set." << std::endl;
            return -1;
        }

        std::cout << "RPC Link: " << globalRpcLink << std::endl;

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
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess)
    {
        std::cerr << "cudaGetDeviceCount failed! Do you have a CUDA-capable GPU installed?" << std::endl;
        return -1;
    }

    auto devices = CudaDevice::getAllDevices();
    std::set<int> usedDevices = parseDeviceList(deviceList, CudaDevice::getAllDevices().size());
    std::ostringstream oss_usedDevices;
    for(auto iter = usedDevices.begin(); iter != usedDevices.end(); ++iter) {
        oss_usedDevices << *iter << ",";
    }
    machineId = getMachineId(oss_usedDevices.str());
    if (!globalWorkerId.empty()) {
        machineId = globalWorkerId;
    }
    std::cout << "Machine ID: " << machineId << std::endl;

    if (!isTestFixedDiff) {
        globalDifficulty = 42069;
        updateDifficulty();
        std::thread difficultyThread(updateDifficultyPeriodically);
        difficultyThread.detach();
    } else {
        std::cout << "Running in TEST MODE with fixed difficulty " << globalDifficulty << std::endl;
    }

    std::thread uploadThread(uploadGpuInfos);
    uploadThread.detach();

    std::thread submitThread(workerThread);
    submitThread.detach();

    Logger logger("log", 1024 * 1024);
    SubmitCallback submitCallback = [&logger](const std::string &hexsalt, const std::string &key, const std::string &hashed_pure, const size_t attempts, const float hashrate) {

        // Report block to platform via MQTT (always, regardless of test mode)
        if (globalPlatformManager && globalPlatformManager->isRunning()) {
            int diff_for_verify = 40404;
            {
                std::lock_guard<std::mutex> lock(mtx);
                diff_for_verify = globalDifficulty;
            }
            Argon2idHasher verifyHasher(1, diff_for_verify, 1, hexsalt, HASH_LENGTH);
            std::string verified_hash = verifyHasher.generateHash(key);
            if (verified_hash.find(hashed_pure) != std::string::npos) {
                globalPlatformManager->onBlockFound(verified_hash, key, "0x" + hexsalt, attempts, hashrate);
            }
        }

        std::function<void()> task = [&logger, hexsalt, key, hashed_pure, attempts, hashrate]() {
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
                {"worker", machineId}
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
                    HttpResponse response = httpClient.HttpPost(globalRpcLink+"/verify", payload, 10000); // 10 seconds timeout
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
                                // PowSubmitter::submitPow(address, key, hashed_data);
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

        if (!isTestFixedDiff) {
            std::lock_guard<std::mutex> lock(mtx_submit);
            taskQueue.push(std::move(task));
        } else {
            std::cout << "Block found (test mode, RPC skipped)." << std::endl;
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

    std::size_t i = 0;
    for (auto &device : devices)
    {
        if(usedDevices.find(i) != usedDevices.end()){
            std::cout << "Device #" << i << ": "
                    << device.getName() << std::endl;
        }
        i++;
    }
    start_time = std::chrono::system_clock::now();

    for (auto deviceIndex : usedDevices) {
        std::thread t(runMiningOnDevice, deviceIndex, submitCallback, statCallback);
        t.detach();
    }

    // Initialize Platform Manager for hashpower marketplace mode
    if (globalPlatformMode) {
        std::vector<gpuInfo> gpuList;
        for (auto idx : usedDevices) {
            gpuInfo gi;
            gi.index = static_cast<int>(idx);
            gi.name = devices[idx].getName();
            size_t freeMem, totalMem;
            cudaSetDevice(idx);
            cudaMemGetInfo(&freeMem, &totalMem);
            gi.memory = static_cast<int>(std::round(static_cast<float>(totalMem) / (1024 * 1024 * 1024)));
            gi.busId = devices[idx].getPicBusId();
            gi.usingMemory = 0;
            gi.temperature = 0;
            gi.hashrate = 0;
            gi.hashCount = 0;
            gpuList.push_back(gi);
        }

        globalPlatformManager = std::make_unique<PlatformManager>(
            globalMqttBroker, globalUserAddress, gpuList);

        if (!globalPlatformManager->start()) {
            std::cerr << RED << "Failed to start PlatformManager. Continuing in self-mining mode." << RESET << std::endl;
            globalPlatformManager.reset();
        } else {
            std::cout << GREEN << "Platform mode enabled. Broker: " << globalMqttBroker << RESET << std::endl;
        }
    }

    app.loglevel(crow::LogLevel::Warning);
    app.signal_clear();
    CROW_ROUTE(app, "/stats")
    ([](){
        auto stats = getGpuStatsJson();
        return stats;
    });
    CROW_ROUTE(app, "/platform/status")
    ([](){
        nlohmann::json result;
        result["platform_mode"] = globalPlatformMode;
        auto ctx = MiningCoordinator::getInstance().getContext();
        result["mining_mode"] = ctx.mode == MiningMode::PLATFORM_MINING ? "platform" : "self";
        if (globalPlatformManager) {
            result["platform_state"] = platformStateToString(globalPlatformManager->getState());
            result["running"] = globalPlatformManager->isRunning();
            auto lease = globalPlatformManager->getLeaseManager().getLease();
            if (lease.has_value()) {
                result["lease_id"] = lease->lease_id;
                result["consumer_id"] = lease->consumer_id;
                result["consumer_address"] = lease->consumer_address;
                result["blocks_found"] = lease->blocks_found;
                result["remaining_sec"] = globalPlatformManager->getLeaseManager().remainingSeconds();
            }
        } else {
            result["platform_state"] = "disabled";
            result["running"] = false;
        }
        return result.dump();
    });
    std::thread serverThread(startServer);
    serverThread.detach();
    if(!donotupload){
        std::thread uploadStatThread(UploadDataPeriodically, 60);
        uploadStatThread.detach();
    }

    while (running)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Clean shutdown of platform manager
    if (globalPlatformManager) {
        globalPlatformManager->stop();
        globalPlatformManager.reset();
    }

    std::cout << std::endl;
    return 0;
}

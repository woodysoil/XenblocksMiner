// MachineIDGetter.cpp
#include "MachineIDGetter.h"
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#include <iphlpapi.h>
#include <windows.h>

#elif defined(__linux__) || defined(__linux)
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#else
#error "OS not supported"
#endif

std::string MachineIDGetter::getMachineId() {
#if defined(_WIN32) || defined(_WIN64)
    ULONG outBufLen = sizeof(IP_ADAPTER_INFO);
    PIP_ADAPTER_INFO pAdapterInfo = (IP_ADAPTER_INFO *)malloc(outBufLen);
    if (GetAdaptersInfo(pAdapterInfo, &outBufLen) == ERROR_BUFFER_OVERFLOW) {
        free(pAdapterInfo);
        pAdapterInfo = (IP_ADAPTER_INFO *)malloc(outBufLen);
    }

    if (GetAdaptersInfo(pAdapterInfo, &outBufLen) == NO_ERROR) {
        for (PIP_ADAPTER_INFO pAdapter = pAdapterInfo; pAdapter != nullptr;
             pAdapter = pAdapter->Next) {
            char macAddr[18] = {};
            sprintf(macAddr, "%02X:%02X:%02X:%02X:%02X:%02X",
                    pAdapter->Address[0], pAdapter->Address[1],
                    pAdapter->Address[2], pAdapter->Address[3],
                    pAdapter->Address[4], pAdapter->Address[5]);
            free(pAdapterInfo);
            return std::string(macAddr);
        }
    }

    if (pAdapterInfo) {
        free(pAdapterInfo);
    }
    throw std::runtime_error("No network adapters found.");

#elif defined(__linux__) || defined(__linux)

    struct ifaddrs *ifap0 = nullptr, *ifap = nullptr;
    std::string macAddress = "";
    if (getifaddrs(&ifap0) >= 0) {
        std::vector<std::string> macAddresses;
        for (ifap = ifap0; ifap != nullptr; ifap = ifap->ifa_next) {
            if (ifap->ifa_addr && ifap->ifa_addr->sa_family == AF_PACKET) {
                int sock = socket(AF_INET, SOCK_DGRAM, 0);
                struct ifreq ifr;
                char mac[18] = {0};

                strncpy(ifr.ifr_name, ifap->ifa_name, IFNAMSIZ - 1);
                if (sock >= 0 && ioctl(sock, SIOCGIFHWADDR, &ifr) >= 0) {
                    unsigned char *hwaddr =
                        (unsigned char *)ifr.ifr_hwaddr.sa_data;
                    sprintf(mac, "%02X:%02X:%02X:%02X:%02X:%02X", hwaddr[0],
                            hwaddr[1], hwaddr[2], hwaddr[3], hwaddr[4],
                            hwaddr[5]);
                    macAddresses.push_back(std::string(mac));
                }
                if (sock >= 0)
                    close(sock);
            }
        }
        freeifaddrs(ifap0);

        std::sort(macAddresses.begin(), macAddresses.end());

        for (const auto &addr : macAddresses) {
            macAddress += addr;
        }
    }

    std::string hostname = getenv("HOSTNAME") ? getenv("HOSTNAME") : "unknown";
    std::string user = getenv("USER") ? getenv("USER") : "unknown";
    std::ifstream file("/proc/cpuinfo");
    std::stringstream buffer;
    if (file) {
        buffer << file.rdbuf();
    }
    std::string cpuInfo = buffer.str();

    return hostname + "_" + user + "_" + cpuInfo + macAddress;

#endif
}

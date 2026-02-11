#include "LocalServer.h"

#include <nlohmann/json.hpp>
#include "StatReporter.h"
#include "MiningCommon.h"
#include "MiningCoordinator.h"
#include "PlatformManager.h"

extern bool globalPlatformMode;
extern std::unique_ptr<PlatformManager> globalPlatformManager;

static crow::SimpleApp s_app;

crow::SimpleApp& getApp() {
    return s_app;
}

void startServer() {
    s_app.port(42069).multithreaded().run();
}

void setupRoutes() {
    s_app.loglevel(crow::LogLevel::Warning);
    s_app.signal_clear();

    CROW_ROUTE(s_app, "/stats")
    ([](){
        auto stats = getGpuStatsJson();
        return stats;
    });

    CROW_ROUTE(s_app, "/platform/status")
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
}

#include "crow_all.h"

int main() {
    crow::SimpleApp app;
    app.loglevel(crow::LogLevel::Warning);

    CROW_ROUTE(app, "/")([](){
        return "Hello, world!";
    });

    app
    app.port(8080).run();
}

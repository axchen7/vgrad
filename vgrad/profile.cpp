#include "profile.h"

#include <thread>

void bar() {
    PROFILE_SCOPE("bar");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void foo() {
    PROFILE_SCOPE("foo");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    bar();
    bar();
}

int main() {
    PROFILE_SCOPE("main");
    foo();
}

#ifndef VGRAD_PROFILE_H_
#define VGRAD_PROFILE_H_

#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

#define PROFILE_SCOPE(label) auto _profile_scope = vgrad::profile::_global_profile_instance.profile_scope(label)
#define PROFILE_NODE *(_profile_scope.enter_scope_node)

namespace vgrad::profile {

using ProfileHookDuration = std::chrono::nanoseconds;
using ProfileHook = std::function<void(ProfileHookDuration duration, std::ostream& os)>;

class ProfileNode {
   public:
    const std::string label;
    ProfileNode* parent;
    std::vector<ProfileNode> children{};
    std::vector<ProfileHook> hooks{};

    ProfileNode(const std::string label, ProfileNode* parent) : label{label}, parent{parent} {
        start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        if (stopped) {
            throw std::runtime_error("ProfileNode " + label + " has already been stopped");
        }
        end = std::chrono::high_resolution_clock::now();
        stopped = true;
    }

    auto duration() const {
        if (!stopped) {
            throw std::runtime_error("ProfileNode " + label + " has not been stopped");
        }
        return end - start;
    }

    void add_hook(ProfileHook hook) { hooks.push_back(hook); }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    bool stopped = false;
};

class AutoScopeProfiler {
   public:
    ProfileNode* enter_scope_node;

    AutoScopeProfiler(ProfileNode* enter_scope_node, const std::function<void()> on_exit_scope)
        : enter_scope_node{enter_scope_node}, on_exit_scope{on_exit_scope} {}
    ~AutoScopeProfiler() { on_exit_scope(); }

   private:
    const std::function<void()> on_exit_scope;
};

class ProfileInstance {
   public:
    ProfileInstance(std::ostream& os) : os{os} {}

    AutoScopeProfiler profile_scope(const std::string label) {
        current->children.emplace_back(label, current);
        current = &current->children.back();
        ProfileNode* enter_scope_node = current;
        return AutoScopeProfiler(enter_scope_node, [this, enter_scope_node]() {
            if (current != enter_scope_node) {
                throw std::runtime_error("Profile scope mismatch");
            }
            current->stop();
            auto parent = current->parent;  // never nullptr because start_of_scope->parent is never nullptr
            current = parent;
        });
    }

    ~ProfileInstance() {
        root.stop();

#ifdef PRINT_PROFILE_ON_EXIT
        print_profile();
#endif
    }

   private:
    std::ostream& os;
    ProfileNode root{"root", nullptr};
    ProfileNode* current = &root;  // never set to nullptr

    void print_profile_rec(const ProfileNode& node, const int depth) const {
        auto duration = node.duration();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

        if (duration_ms > std::chrono::milliseconds{0}) {
            for (int i = 0; i < depth; i++) {
                os << "  ";
            }
            os << node.label << ": " << duration_ms;

            for (const auto& hook : node.hooks) {
                os << " | ";
                hook(duration, os);
            }
            os << "\n";
        }

        for (const auto& child : node.children) {
            print_profile_rec(child, depth + 1);
        }
    }

    void print_profile() const {
        os << "\nProfile results:\n----------------\n";
        if (current != &root) {
            throw std::runtime_error("Still in a profile scope: " + current->label);
        }
        print_profile_rec(root, 0);
        os << "----------------\n\n";
    }
};

ProfileInstance _global_profile_instance{std::cout};

}  // namespace vgrad::profile

#endif  // VGRAD_PROFILE_H_
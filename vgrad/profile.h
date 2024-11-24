#ifndef VGRAD_PROFILE_H_
#define VGRAD_PROFILE_H_

#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

namespace vgrad::profile {

using Granularity = std::chrono::milliseconds;

class ProfileNode {
   public:
    const std::string label;
    ProfileNode* parent;
    std::vector<ProfileNode> children{};

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
        return std::chrono::duration_cast<Granularity>(end - start);
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    bool stopped = false;
};

class AutoScopeProfiler {
   public:
    AutoScopeProfiler(const std::function<void()> on_exit_scope) : on_exit_scope{on_exit_scope} {}
    ~AutoScopeProfiler() { on_exit_scope(); }

   private:
    const std::function<void()> on_exit_scope;
};

class ProfileInstance {
   public:
    AutoScopeProfiler profile_scope(const std::string label) {
        current->children.emplace_back(label, current);
        current = &current->children.back();
        return AutoScopeProfiler([this] {
            current->stop();
            auto parent = current->parent;
            if (parent == nullptr) {
                throw std::runtime_error("ProfileNode " + current->label + " has no parent");
            }
            current = parent;
        });
    }

    ~ProfileInstance() {
        root.stop();
        print_profile();
    }

   private:
    ProfileNode root{"root", nullptr};
    ProfileNode* current = &root;  // never set to nullptr

    void print_profile_rec(const ProfileNode& node, const int depth) const {
        for (int i = 0; i < depth; i++) {
            std::cout << "  ";
        }
        std::cout << node.label << ": " << node.duration() << "\n";
        for (const auto& child : node.children) {
            print_profile_rec(child, depth + 1);
        }
    }

    void print_profile() const {
        if (current != &root) {
            throw std::runtime_error("Still in a profile scope: " + current->label);
        }
        print_profile_rec(root, 0);
    }
};

ProfileInstance _global_profile_instance;

#define PROFILE_SCOPE(label) auto _profile_scope = _global_profile_instance.profile_scope(label)

}  // namespace vgrad::profile

#endif  // VGRAD_PROFILE_H_
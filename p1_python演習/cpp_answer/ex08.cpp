#include<iostream>
#include<string>
#include<map>
#include<vector>
#include<array>
#include<algorithm>

std::vector<std::string> split(const std::string& target, const std::vector<char>& split_char) {
    std::vector<std::string> result;
    int pre = 0;
    for (int i = 0; i < target.size(); i++) {
        if (std::find(split_char.begin(), split_char.end(), target[i]) != split_char.end()) {
            //分割
            if (pre != i) {
                result.push_back(target.substr(pre, i - pre));
            }
            pre = i + 1;
        }
    }
    return result;
}

int main() {
    std::string str = "Hi He Lead Because Boron Could Not Oxidize Flourine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.";
    constexpr std::array<int, 9> one{ 1, 5, 6, 7, 8, 9, 15, 16, 19 };
    std::map<std::string, int> mp;
    std::vector<char> split_char{ ' ', ',', '.' };
    auto result = split(str, split_char);
    for (int i = 0; i < result.size(); i++) {
        if (std::find(one.begin(), one.end(), i + 1) != one.end()) {
            mp[result[i].substr(0, 1)] = i + 1;
        } else {
            mp[result[i].substr(0, 2)] = i + 1;
        }
    }

    for (auto p : mp) {
        std::cout << p.first << ":" << p.second << std::endl;
    }
}

#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<fstream>
#include<cmath>

double idf(const std::string& term, const std::vector<std::vector<std::string>>& docs) {
    int df = 0;
    for (const auto& doc : docs) {
        for (const auto& w : doc) {
            if (w == term) {
                df++;
                break;
            }
        }
    }
    return std::log10((double)docs.size() / df) + 1;
}

int main() {
    std::vector<std::vector<std::string>> docs {
        { "リンゴ", "リンゴ" },
        { "リンゴ", "レモン" },
        { "レモン", "ミカン" }
    };
    std::vector<std::string> terms {
        "リンゴ", "レモン", "ミカン"
    };

    for (const auto& w : terms) {
        std::cout << "idf(" << w << ") = " << idf(w, docs) << std::endl;
    }
}

#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<fstream>

double tf(const std::string& term, const std::vector<std::string>& doc) {
    int num = 0;
    for (const auto& w : doc) {
        if (w == term) {
            num++;
        }
    }
    return (double)num / doc.size();
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

    for (const auto& v : docs) {
        for (const auto& w : terms) {
            std::cout << "tf(" << w << ", [ ";
            for (const auto& t : v) {
                std::cout << t << ", ";
            }
            printf("]) = %.3f  ", tf(w, v));
        }
        printf("\n");
    }
}

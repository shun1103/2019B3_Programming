#include<iostream>
#include<string>
#include<vector>
#include<set>
#include<fstream>

int main() {
    std::vector<std::vector<std::string>> docs;
    std::set<std::string> terms;

    std::ifstream ifs("data.txt");
    std::string str;
    while (std::getline(ifs, str)) {
        std::vector<std::string> v;
        int pre = 0;
        while (true) {
            auto st = str.find("と", pre);
            if (st == std::string::npos) {
                st = str.size();
            }
            v.push_back(str.substr(pre, st - pre));
            if (st == str.size()) {
                break;
            }
            pre = st + 3;
        }
        docs.push_back(v);
    }

    for (int i = 0; i < docs.size(); i++) {
        for (int j = 0; j < docs[i].size(); j++) {
            std::cout << docs[i][j] << " \n"[j == docs[i].size() - 1];
        }
    }

    for (const auto& v : docs) {
        for (const auto& w : v) {
            terms.insert(w);
        }
    }

    for (const auto& e : terms) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

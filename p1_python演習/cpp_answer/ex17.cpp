#include"nlp/tf_idf.hpp"
#include"nlp/cosine_sim.hpp"
#include<iostream>

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

    auto tf_idf_mat = tf_idf(terms, docs);

    std::vector<std::vector<double>> cosine_sim_mat(docs.size(), std::vector<double>(docs.size()));
    for (int i = 0; i < docs.size(); i++) {
        for (int j = 0; j < docs.size(); j++) {
            cosine_sim_mat[i][j] = cosine_sim(tf_idf_mat[i], tf_idf_mat[j]);
        }
    }

    for (int i = 0; i < docs.size(); i++) {
        printf("[");
        for (int j = 0; j < docs.size(); j++) {
            printf("%.10f%s", cosine_sim_mat[i][j], j == docs.size() - 1 ? "]\n" : "  ");
        }
    }
}

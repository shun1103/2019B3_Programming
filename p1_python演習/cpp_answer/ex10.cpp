#include<iostream>
#include<string>
#include<vector>

int main(int argc, char* argv[]) {
    std::string str_n(argv[1]);
    int n = std::stoi(str_n);

    std::vector<std::vector<std::string>> word_n_gram;
    std::vector<std::string> char_n_gram;

    std::vector<std::string> targets;
    std::string str_target;
    for (int i = 2; i < argc; i++) {
        std::string str(argv[i]);
        targets.push_back(str);
        str_target += str;
    }

    for (int i = 0; i <= targets.size() - n; i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < n; j++) {
            tmp.push_back(targets[i + j]);
        }

        word_n_gram.push_back(tmp);
    }

    for (int i = 0; i <= str_target.size() - n; i++) {
        char_n_gram.push_back(str_target.substr(i, n));
    }

    std::cout << "単語" << n << "-gram: ";
    for (const auto& v : word_n_gram) {
        std::cout << "[";
        for (int i = 0; i < v.size(); i++) {
            std::cout << (i != 0 ? ", " : "") << v[i];
        }
        std::cout << "], ";
    }
    std::cout << std::endl;

    std::cout << "文字" << n << "-gram: ";
    for (const auto& v : char_n_gram) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
}

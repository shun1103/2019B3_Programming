#include<iostream>
#include<string>
#include<vector>

int main() {
    std::string str = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.";
    std::vector<int> v;

    int num = 0;
    for (char c : str) {
        switch (c) {
            case ' ':
            case ',':
            case '.':
                v.push_back(num);
                num = 0;
                break;
            default:
                num++;
        }
    }

    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " \n"[i == v.size() - 1];
    }
}

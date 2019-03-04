#include<iostream>
#include<vector>

int main() {
    std::vector<int> v;
    for (int i = 1; i <= 10; i++) {
        v.push_back(i * i);
    }
    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " \n"[i == v.size() - 1];
    }
}

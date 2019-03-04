#include<iostream>
#include<vector>
#include<algorithm>

int main() {
    std::vector<int> v(10);

    std::generate(v.begin(), v.end(), []() {
        static int i = 0;
        i++;
        return i * i;
    });

    for (int i = 0; i < v.size(); i++) {
        std::cout << v[i] << " \n"[i == v.size() - 1];
    }
}

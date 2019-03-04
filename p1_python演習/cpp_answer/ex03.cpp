#include<iostream>
#include<string>
#include<algorithm>

int main() {
    std::string str = "stressed";
    std::reverse(str.begin(), str.end());
    std::cout << str << std::endl;
}

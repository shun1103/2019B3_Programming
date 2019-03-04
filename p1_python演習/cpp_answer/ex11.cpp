#include<iostream>
#include<string>
#include<fstream>

int main() {
    std::ifstream ifs("data.txt");
    std::string str;
    while (std::getline(ifs, str)) {
        std::cout << str << std::endl;
    }
}

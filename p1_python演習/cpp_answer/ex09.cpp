#include<iostream>
#include<vector>
#include<algorithm>
#include<random>

struct ThreeByteChar {
    char c[3];
};

int main(int argc, char* argv[]) {
    std::mt19937 engine(2);

    for (int i = 1; i < argc; i++) {
        std::vector<ThreeByteChar> v;
        for (int j = 0; ; j += 3) {
            bool finish = false;
            for (int k = 0; k < 3; k++) {
                if (argv[i][j + k] == '\0') {
                    finish = true;
                }
            }
            if (finish) {
                break;
            }
            ThreeByteChar tmp;
            tmp.c[0] = argv[i][j];
            tmp.c[1] = argv[i][j + 1];
            tmp.c[2] = argv[i][j + 2];
            v.push_back(tmp);
        }

        if (v.size() >= 3) {
            std::shuffle(v.begin() + 1, v.end() - 1, engine);
        }
        for (auto& tbc : v) {
            std::string buf{ tbc.c[0], tbc.c[1], tbc.c[2] };
            std::cout << buf;
        }
        std::cout << " ";
    }
    std::cout << std::endl;
}

#include<iostream>
#include<string>

int main() {
    std::string str = "パタトクカシーー";
    std::string ans[2];

    //utf-8ではカタカナは3byteでエンコードされる
    //きちんと書くならば既存の文字列操作ライブラリを利用するべき
    for (int i = 0; i < str.size(); i += 3) {
        ans[(i / 3) % 2] += str.substr(i, 3);
    }
    std::cout << ans[0] << std::endl;
    std::cout << ans[1] << std::endl;
}

#include<vector>
#include<cmath>

double cosine_sim(const std::vector<double>& x1, const std::vector<double>& x2) {
    double ip = 0.0, x1_norm = 0.0, x2_norm = 0.0;
    for (int i = 0; i < x1.size(); i++) {
        ip += x1[i] * x2[i];
        x1_norm += x1[i] * x1[i];
        x2_norm += x2[i] * x2[i];
    }
    x1_norm = std::sqrt(x1_norm);
    x2_norm = std::sqrt(x2_norm);
    return ip / (x1_norm * x2_norm);
}

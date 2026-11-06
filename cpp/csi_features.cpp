// CrossGuard-Fi: deterministic C++17 feature extraction for CSI amplitude windows.
// Input: one CSV sample per row, fixed channel count. Output: mean and stddev/channel.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct RunningStat {
    std::size_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
    void add(double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (value - mean);
    }
    double stddev() const {
        return count ? std::sqrt(m2 / static_cast<double>(count)) : 0.0;
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: csi_features <channel_count> < window.csv\n";
        return 2;
    }
    try {
        const std::string arg = argv[1];
        std::size_t consumed = 0;
        const long parsed = std::stol(arg, &consumed);
        if (consumed != arg.size() || parsed <= 0 || parsed > 10000) {
            throw std::invalid_argument("invalid channel count");
        }
        const auto channels = static_cast<std::size_t>(parsed);
        std::vector<RunningStat> stats(channels);
        std::string line;
        std::size_t rows = 0;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::stringstream row(line);
            std::string cell;
            std::size_t col = 0;
            while (std::getline(row, cell, ',')) {
                if (col >= channels) throw std::runtime_error("too many channels");
                std::size_t end = 0;
                const double value = std::stod(cell, &end);
                if (cell.find_first_not_of(" \t\r", end) != std::string::npos ||
                    !std::isfinite(value)) throw std::runtime_error("invalid numeric sample");
                stats[col++].add(value);
            }
            if (col != channels) throw std::runtime_error("wrong channel count");
            ++rows;
        }
        if (rows == 0) throw std::runtime_error("empty window");
        std::cout << "channel,mean,stddev\n" << std::fixed << std::setprecision(6);
        for (std::size_t col = 0; col < channels; ++col) {
            std::cout << col << ',' << stats[col].mean << ',' << stats[col].stddev() << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Input error: " << e.what() << '\n';
        return 1;
    }
}

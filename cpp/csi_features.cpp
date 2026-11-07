// CrossGuard-Fi C++17: stable, deterministic per-channel CSI amplitude statistics.
// Headerless CSV on stdin: one time sample per row, exactly N numeric channels.
// Output: channel,mean,stddev (population sigma). No external C++ dependencies.
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
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

static std::size_t parse_channel_count(const std::string& argument) {
    if (argument.empty() || argument.front() == '-') {
        throw std::invalid_argument("invalid channel count");
    }
    std::size_t consumed = 0;
    const auto parsed = std::stoul(argument, &consumed, 10);
    if (consumed != argument.size() || parsed == 0 || parsed > 10000) {
        throw std::invalid_argument("channel count must be between 1 and 10000");
    }
    return static_cast<std::size_t>(parsed);
}

static double parse_value(const std::string& cell) {
    if (cell.empty()) {
        throw std::runtime_error("empty CSV cell");
    }
    std::size_t consumed = 0;
    const double value = std::stod(cell, &consumed);
    if (cell.find_first_not_of(" \t\r", consumed) != std::string::npos ||
            !std::isfinite(value)) {
        throw std::runtime_error("invalid or non-finite CSI sample");
    }
    return value;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: csi_features <channel_count> < window.csv\n";
        return 2;
    }
    try {
        const auto channels = parse_channel_count(argv[1]);
        std::vector<RunningStat> statistics(channels);
        std::string line;
        std::size_t rows = 0;
        while (std::getline(std::cin, line)) {
            if (line.empty()) throw std::runtime_error("blank CSV row");
            if (line.size() > 1'000'000) throw std::runtime_error("oversized CSV row");
            if (line.back() == ',') throw std::runtime_error("empty trailing CSV cell");
            std::stringstream row(line);
            std::string cell;
            std::size_t col = 0;
            while (std::getline(row, cell, ',')) {
                if (col >= channels) throw std::runtime_error("too many channels");
                statistics[col++].add(parse_value(cell));
            }
            if (col != channels) throw std::runtime_error("wrong channel count");
            ++rows;
        }
        if (rows == 0) throw std::runtime_error("empty window");
        std::cout << "channel,mean,stddev\n" << std::fixed << std::setprecision(6);
        for (std::size_t col = 0; col < channels; ++col) {
            std::cout << col << ',' << statistics[col].mean << ','
                      << statistics[col].stddev() << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "CSI input error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

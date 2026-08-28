#include "n8_binary_format.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using Record = n8fmt::HardRecord;
using Out = n8fmt::CertificateRecord;
using u64 = std::uint64_t;

namespace {

template<class T>
void read_exact(std::istream& in, T& value, const char* what) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error(std::string("cannot read ") + what);
}

template<class T>
std::vector<T> read_records(std::istream& in, u64 count, const char* what) {
    if (count > static_cast<u64>(std::numeric_limits<std::size_t>::max() / sizeof(T))) {
        throw std::runtime_error(std::string(what) + " count does not fit in memory");
    }
    std::vector<T> result(static_cast<std::size_t>(count));
    if (!result.empty()) {
        in.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(result.size() * sizeof(T)));
        if (!in) throw std::runtime_error(std::string("truncated ") + what + " payload");
    }
    char extra = 0;
    if (in.read(&extra, 1)) throw std::runtime_error(std::string("trailing bytes in ") + what);
    return result;
}

std::vector<Record> read_hard(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open hard file: " + path);
    u64 magic = 0, count = 0;
    read_exact(in, magic, "hard magic");
    read_exact(in, count, "hard count");
    if (magic != n8fmt::hard_magic) throw std::runtime_error("bad hard-file magic");
    auto records = read_records<Record>(in, count, "hard file");
    for (const auto& r : records) {
        if (r.reserved != 0) throw std::runtime_error("nonzero hard-record reserved byte");
    }
    return records;
}

std::vector<Out> read_certificates(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open certificate file: " + path);
    u64 count = 0;
    read_exact(in, count, "certificate count");
    auto records = read_records<Out>(in, count, "certificate file");
    for (const auto& o : records) {
        if (o.r.reserved != 0) throw std::runtime_error("nonzero embedded hard-record reserved byte");
        if (!std::all_of(o.reserved.begin(), o.reserved.end(), [](auto x) { return x == 0; })) {
            throw std::runtime_error("nonzero certificate reserved bytes");
        }
        if (o.type > 1) throw std::runtime_error("unknown certificate type");
    }
    return records;
}

u64 read_failure_count(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open failure file: " + path);
    u64 count = 0;
    read_exact(in, count, "failure count");
    if (count > static_cast<u64>(std::numeric_limits<std::size_t>::max() / sizeof(Out))) {
        throw std::runtime_error("failure count does not fit in memory");
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    const auto expected = static_cast<std::streamoff>(sizeof(u64) + count * sizeof(Out));
    if (size != expected) throw std::runtime_error("failure-file length mismatch");
    return count;
}

auto id(const Record& r) {
    return std::tuple{r.key, r.h, r.k, r.Bmask, r.flags};
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4) {
            std::cerr << "usage: record_checker hard.bin cert.bin [fail.bin]\n";
            return 2;
        }
        const auto hard = read_hard(argv[1]);
        const auto cert = read_certificates(argv[2]);
        if (hard.size() != cert.size()) {
            throw std::runtime_error("count mismatch hard=" + std::to_string(hard.size()) +
                                     " cert=" + std::to_string(cert.size()));
        }

        using Id = decltype(id(Record{}));
        std::vector<Id> a, b;
        a.reserve(hard.size());
        b.reserve(cert.size());
        for (const auto& r : hard) a.push_back(id(r));
        for (const auto& o : cert) b.push_back(id(o.r));
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) throw std::runtime_error("record-set mismatch");
        if (std::adjacent_find(a.begin(), a.end()) != a.end()) {
            throw std::runtime_error("duplicate hard record");
        }
        if (std::adjacent_find(b.begin(), b.end()) != b.end()) {
            throw std::runtime_error("duplicate certificate record");
        }

        u64 failures = 0;
        if (argc == 4) {
            failures = read_failure_count(argv[3]);
            if (failures != 0) throw std::runtime_error("failure file is nonempty");
        }
        std::cout << "PASS hard_records=" << hard.size()
                  << " certificate_records=" << cert.size()
                  << " failures=" << failures << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: " << ex.what() << "\n";
        return 1;
    }
}

#include "n8_binary_format.hpp"
#include <cstddef>
#include <iostream>

int main() {
    using namespace n8fmt;
    static_assert(offsetof(HardRecord, key) == 0);
    static_assert(offsetof(HardRecord, h) == 8);
    static_assert(offsetof(HardRecord, k) == 12);
    static_assert(offsetof(HardRecord, Bmask) == 13);
    static_assert(offsetof(HardRecord, flags) == 14);
    static_assert(offsetof(HardRecord, reserved) == 15);
    static_assert(offsetof(HardRecord, eps) == 16);
    static_assert(offsetof(CertificateRecord, r) == 0);
    static_assert(offsetof(CertificateRecord, committee) == 24);
    static_assert(offsetof(CertificateRecord, deficit) == 26);
    static_assert(offsetof(CertificateRecord, type) == 27);
    static_assert(offsetof(CertificateRecord, reserved) == 28);
    static_assert(offsetof(CertificateRecord, sg) == 32);
    static_assert(offsetof(CertificateRecord, coal) == 40);
    static_assert(offsetof(CertificateRecord, allcm) == 48);
    std::cout << "PASS hard_record_size=" << sizeof(HardRecord)
              << " certificate_record_size=" << sizeof(CertificateRecord)
              << " endian=little ieee754_binary64=1\n";
}

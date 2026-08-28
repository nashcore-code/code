#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

// Canonical binary layouts used by the eight-row artifact.
//
// Files are little-endian and use IEEE-754 binary64.  The current C++ readers
// deliberately reject other host representations at compile time rather than
// silently interpreting a proof record incorrectly.  Python merge/inspection
// tools use explicit '<' struct formats and are host-independent.
namespace n8fmt {

using u8 = std::uint8_t;
using i8 = std::int8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static_assert(std::endian::native == std::endian::little,
              "the C++ proof-record readers currently require a little-endian host");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "the proof-record format requires IEEE-754 binary64");
static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8);

inline constexpr u64 hard_magic = 0x3843454c4c533031ULL; // ASCII-ish "8CELLS01"
inline constexpr std::array<char, 8> m6_certificate_magic = {'M','6','C','E','R','T','0','1'};

// Header: uint64 magic, uint64 record_count.  Each record below is 24 bytes.
struct HardRecord {
    u64 key{};      // canonical row-pattern key
    u32 h{};        // eight 3-bit floor coordinates, voter 0 in low bits
    u8 k{};         // residual committee budget
    u8 Bmask{};     // usable/deficit voter mask
    u8 flags{};     // bit 0 historically records Bmask==0
    u8 reserved{};  // must be zero
    double eps{};   // proposal diagnostic; scanners currently store 1.0
};
static_assert(std::is_standard_layout_v<HardRecord> && std::is_trivially_copyable_v<HardRecord>);
static_assert(sizeof(HardRecord) == 24);

// Header: uint64 record_count.  Each record below is 64 bytes.
// The four reserved bytes make the former ABI padding explicit and deterministic.
struct CertificateRecord {
    HardRecord r{};
    u16 committee{};             // fixed committee, or E-mask for adaptive records
    i8 deficit{};                // fixed deficit voter; -1 for adaptive records
    u8 type{};                   // 0=fixed, 1=adaptive
    std::array<u8, 4> reserved{};
    double sg{};   // proposal-only diagnostic
    double coal{};          // proposal-only diagnostic
    std::array<u16, 8> allcm{};  // adaptive committee indexed by voter
};
static_assert(std::is_standard_layout_v<CertificateRecord> &&
              std::is_trivially_copyable_v<CertificateRecord>);
static_assert(sizeof(CertificateRecord) == 64);

} // namespace n8fmt

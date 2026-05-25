#include "blake2b.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

namespace {

using Digest = std::array<std::uint8_t, Blake2b::OUT_BYTES>;

Digest digestOneShot(std::string_view prefix, std::string_view suffix)
{
    Digest out{};
    Blake2b blake;
    blake.init(out.size());
    blake.update(prefix.data(), prefix.size());
    blake.update(suffix.data(), suffix.size());
    blake.final(out.data(), out.size());
    return out;
}

Digest digestFromCopiedState(std::string_view prefix, std::string_view suffix)
{
    Blake2b base;
    base.init(Blake2b::OUT_BYTES);
    base.update(prefix.data(), prefix.size());

    Blake2b copied = base;
    copied.update(suffix.data(), suffix.size());

    Digest out{};
    copied.final(out.data(), out.size());
    return out;
}

bool sameDigest(const Digest& lhs, const Digest& rhs)
{
    return std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0;
}

} // namespace

int main()
{
    constexpr std::string_view prefix =
        "argon2id-xen-prefix-state-copy-diagnostic-prefix-crosses-one-blake2b-block-";
    constexpr std::string_view suffix_a = "candidate-key-material-a";
    constexpr std::string_view suffix_b = "candidate-key-material-b";

    const Digest direct_a = digestOneShot(prefix, suffix_a);
    const Digest copied_a = digestFromCopiedState(prefix, suffix_a);
    const Digest direct_b = digestOneShot(prefix, suffix_b);
    const Digest copied_b = digestFromCopiedState(prefix, suffix_b);

    const bool copy_matches_direct = sameDigest(direct_a, copied_a) && sameDigest(direct_b, copied_b);
    const bool copied_states_are_independent = !sameDigest(copied_a, copied_b);
    const bool ok = copy_matches_direct && copied_states_are_independent;

    std::cout
        << "{\"ok\":" << (ok ? "true" : "false")
        << ",\"copy_matches_direct\":" << (copy_matches_direct ? "true" : "false")
        << ",\"copied_states_are_independent\":" << (copied_states_are_independent ? "true" : "false")
        << "}\n";

    return ok ? 0 : 1;
}

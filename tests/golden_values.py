"""Reference outputs. Change them only on purpose (new weights, new sampler semantics) and say why in the commit.

RANDOM_FIRST/RANDOM_SECOND match the xoshiro256** reference implementation seeded with SplitMix64(42). The other
values are identical to those of the original C# prototype, which confirms the port kept the same semantics.
"""

RANDOM_FIRST = 1546998764402558742
RANDOM_SECOND = 6990951692964543102
SYSTEM_FINGERPRINT = "fp_aff631e39a75"
GREEDY_FINGERPRINT = "706ce9aaacf0e839e9aa2f14318b1b7f72028e7cb0e23ecd7c8431008ff35e89"
SAMPLED_FINGERPRINT = "cb6d23ee7267e79f4bcbb50f965626e05eb2f8219a8c26a6155f9e30f20b9cc6"

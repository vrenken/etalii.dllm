"""Reference outputs. Change them only on purpose (new weights, new sampler semantics) and say why in the commit.

RANDOM_FIRST/RANDOM_SECOND match the xoshiro256** reference implementation seeded with SplitMix64(42). The other
values are identical to those of the original C# prototype, which confirms the port kept the same semantics.
"""

RANDOM_FIRST = 1546998764402558742
RANDOM_SECOND = 6990951692964543102
SYSTEM_FINGERPRINT = "fp_aff631e39a75"
GREEDY_FINGERPRINT = "706ce9aaacf0e839e9aa2f14318b1b7f72028e7cb0e23ecd7c8431008ff35e89"
SAMPLED_FINGERPRINT = "cb6d23ee7267e79f4bcbb50f965626e05eb2f8219a8c26a6155f9e30f20b9cc6"

# Tensor.fingerprint() of the Phase 1 kernel outputs on fixed inputs (tests/test_kernels.py::kernel_outputs).
KERNEL_FINGERPRINTS = {
    "linear": "e66492b8888156fa615456c67a59874296d456f3810f75fe94d342451ad651f6",
    "matmul": "a9d0e49cc8501640bbdd7105bee0ba175b0478005c91015d10281fee9e8cba15",
    "rms_norm": "5ad77034e13064b9a736234e7e40cb355cc42fd2065f4df0c5d7240db5aaf564",
    "silu": "67b9be83fdacaf9b609783a19a56e6ecd5f55917cfe7a7cd901afc5bbde2711e",
    "gelu": "bce07a6466ec14107d524cdb76b5ca92d72314e4dd6439b5e80dde39faaec9c2",
    "gelu_tanh": "a000c1fda83bd19d6b6932ae3e97b3451f5fd1bdfbef6cdc4fb38a82c6bd5647",
    "rope": "9cd1ae74870656556f13dca8726d1e4430a1cc68934c2b5e06b5529ccfd77158",
    "attention": "c7e16ac447a7dcb06878fa1d122a7d19370d1d95502a21027bf2551f4a6ba18d",
}

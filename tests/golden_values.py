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
    # Phase 6: Q8_0 quantised linear (weights and activations in 32-blocks, exact int32 sums).
    "linear_q8": "af1ce52d09c0a94215a627172f03c4043ca982583fe4d083dd98e9dd5ee00407",
}

# Fingerprint of tests/model_fixtures.py's tiny bf16 checkpoint imported to model.dllm (format version 1). It pins
# the container layout and the exact bf16 -> fp32 conversion; the HF and GGUF imports of those weights both match it.
TINY_IMPORT_FINGERPRINT = "1585fbbfe689fd0d52dc36a4b3668f3af07babda2f72c8bfd33d9849dbabc71e"

# fingerprint() of the decoder's next-token logits for tests/test_transformer.py::PROMPT on the tiny imported models.
TINY_LOGITS_FINGERPRINT = {
    "gemma2": "5b4d9892ba2957acd7f08e7388e886d27040f6b0a3956cfdd360f50959a58f3f",
    "gemma3": "10feca605e2514c582b6d23f28e033b997a9b404f730cd9e2baaec9a37c9eb60",
    "granite": "d899150301a14e954bf92d5f768c0d74ca8fba8b7c60ed433a0e90e89704c5ca",
    "llama": "12a0878056708099f139e39c6948153b7ba152960259846ab9caa940863e1e29",
    "mistral": "a03b36df3ec07340f1e54d9d2150a77afd8e08a261977d61d7d51c59248e075f",
    "olmo2": "4ad8a7f8864541c90f8b7b358e0b7bfae11e966cdde3b3246f331a4f1916a49e",
    "phi3": "6c0319271a7751102f7c5942a64978b1b412a4e474967ef425feec341e242845",
    "qwen2": "9d284fb62e7360dc835fcd1dd9cac2b54c6bff5c66744d980c02025039330d0f",
    "qwen3": "be9c8d225e709f628a67fb3114a81fe94f4c61456e9228869171dbf0c6e3aefc",
}

# Phase 3. fingerprint() of the concatenated gradients (tensors in natural name order) of the summed next-token loss
# for tests/test_training.py::TOKENS on the tiny imported models.
GRADIENT_FINGERPRINT = {
    "llama": "d7c7892dd31fec99e21991a1555dae2e6a431cf8a7e0bef1857e9b3d16cca5c5",
    "mistral": "c458fd0ed0194d5bcef7b150ca5fa3b853b589d3c4c02e44a763faefe19e6db6",
    "qwen2": "59dd1f723fda011ecefaa158fbb1557c05ddc03a97cbadae70d3c358c5081cf3",
    "qwen3": "86666c63aea685d30b8f66d665d7c221071c4b97c13213d1588718c8bd18cbd8",
}

# Fingerprint of the model.dllm exported after tests/test_training.py::RUN (6 AdamW steps on the ASCII data).
FINETUNE_FINGERPRINT = {
    "llama": "8cce2d4c31c0db30903367d948b48a2cac91e0058659f839619a06577c2db5d8",
    "mistral": "eb8535983feda9d1a8c1627c194e8fc97b3e970be901364db4939eca08b66ba1",
    "qwen2": "557ffcff23dfc853308c73c68534cef212ee3873741fd140c4978fad806a2964",
    "qwen3": "bd00cb4c7be0ebba8bc67300af300ea28054a8147f74480b788e7aa131d02514",
}

# Phase 8. Fingerprint of the model.dllm exported after tests/test_lora.py::RUN (6 AdamW steps on rank-2 LoRA
# adapters of every linear layer): the base with the trained adapters merged in.
LORA_FINETUNE_FINGERPRINT = {
    "llama": "05fd6d263be7b2f600637127b9f425ac5a699c51f4b5622eec8f50c888ccbedc",
    "qwen3": "bb2e772b7a102b02be51cb894da3676a2c557956f156f5be5c30ea7a93ba9799",
}

# Phase 4. fingerprint() of the tokens of tests/test_generation.py's schema-constrained chat (placeholder model,
# temperature 0.8, seed 11), and of the placeholder model's mean-pooled embedding of "embed this text".
CONSTRAINED_FINGERPRINT = "979b6d7e752cd5b30ba187fd03c7919367f6494b05b0cdb537f0e12eff149c61"
EMBEDDING_FINGERPRINT = "6cea233a8a650a422c82e849cd5d0c77d223c38d1e9b92f3d84603a3ba4db177"

# Phase 10 (#100): `dllm verify` with the placeholder model; the same on every machine.
VERIFY_FINGERPRINT = "3ffd534d1fadce8a267c510806baa2fc"

# Phase 2 real models (tests/test_reference_models.py): the import fingerprint of the pinned checkpoint, and
# GenerationResult.fingerprint of the greedy chat answer to test_reference_models.CHAT. Phase 10 (#99) adds the
# sampled answer (SAMPLED) and the float32 bits of the last-position logits of PROMPTS (float32 and Q8_0 weights);
# they must be the same on every release platform, SIMD path and device.
REFERENCE_MODEL_FINGERPRINTS = {
    "smollm2": {
        "import": "2e4d93db18c1ce202f44a8fd4da1333532827e2b84b7979ddcba6c918bfc132f",
        "chat": "cb555f4a6519d8b67bc7e1c304b08c0c13852cea297e0fdc644b0b77f70a8f1d",
        "sampled": "5f3620a368decf6e7ad82105879f8ee37cc346961a2332298f700144a3eec488",
        "logits": "939f4dab80703cf2dcfff45656a84cc8fbd6602a8a9c3e6d75ada13ce0c5a5fb",
        "logits_q8_0": "000765286f26f9e922fc23f5dfcbae1b7b53b5574525015b2a956b272252b3f9",
    },
    "qwen2.5": {
        "import": "9bf78203fabbc1d93a5f387456ff1d755d7fef36396351c911f51656c898eb99",
        "chat": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "sampled": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",  # the same answer
        "logits": "cafb17dbac5ba9251af7a494246a70f258d6de372bab1a273897f07e74993404",
        "logits_q8_0": "e599b1133d04f3c72c0e6e32c24f78c9ea4d67db3424e7e8f0f54c05aa6fa831",
    },
    "qwen2.5-1.5b": {
        "import": "626d11f6abd38e28450448eae2574b29de3212825a143f37bdbf145673e8556d",
        # The same answer tokens as Qwen2.5-0.5B ("The capital of France is Paris."), so the same fingerprint.
        "chat": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "sampled": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "logits": "a4eff454791a62e4cfe01c24bd57f9af78809387ae9d699ecf162eeb1007a0a4",
        "logits_q8_0": "fc6ded7859eb616f4baa45870d5e3ad3af9f45757378f288db62d368986b3a73",
    },
    "qwen3": {  # greedy chat rendered with enable_thinking=False
        "import": "c34b4666335e7e0dbe110ee9b9fc78936b84391a57a420688fd78036ad6df5b7",
        "chat": "109ee1a8513bf1dad8a34b0483a0f3045fe18cbf7c03820e4e359cc20912201c",
        "sampled": "109ee1a8513bf1dad8a34b0483a0f3045fe18cbf7c03820e4e359cc20912201c",  # the same answer
        "logits": "54400823b346fbacdb126d5ccca19e12f7e8d3c0d04a94ee546abfa840d1523a",
        "logits_q8_0": "21cd894f2d19253192801ec012209624721e1e401bf530b63d39c7e79900eb5a",
    },
    "tinyllama": {
        "import": "b02b4f3085396f3a0b20e07f5b875c9e4a0fbf996dcbc954953bc7d7346b4a4e",
        "chat": "12608d6f03ed2ea7892990ed181b39a3d0df693637f168ca34008af1403d4bc3",
        "sampled": "12608d6f03ed2ea7892990ed181b39a3d0df693637f168ca34008af1403d4bc3",
        "logits": "1b40cd4dc37c6179feee4abb69e05b42edc3468200e98390685965d893241a1b",
        "logits_q8_0": "5474bab37def485b16798648b8fe338db268e03f1b6544dcb98bc861ff976f53",
    },
    "olmo2": {
        "import": "183baa6ee6dcc08d855a34f1531b7e5cd1d918c21c6e8bef4cebde7ec675d118",
        "chat": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "sampled": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "logits": "41db902b34618b95d3c185d772cf1f10fa58ee3fbd4094a6fb257252528eb5ba",
        "logits_q8_0": "114b3719f3601bfe781d74af647bb13de3345b6fe9a43b11d49825c6ad461b74",
    },
}

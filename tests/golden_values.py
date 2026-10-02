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
# The whole model.dllm of that import: the same bytes on every platform (Phase 17).
TINY_IMPORT_FILE_SHA256 = "a5a1dee96237d9eb18af573ae679a2e43265aac500be85b721e29f8068de4362"

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
# sampled answer (SAMPLED) and the float32 bits of the last-position logits of PROMPTS (float32 and Q8_0 weights;
# Q4_0 since Phase 14); they must be the same on every release platform, SIMD path and device.
REFERENCE_MODEL_FINGERPRINTS = {
    "smollm2": {
        "import": "2e4d93db18c1ce202f44a8fd4da1333532827e2b84b7979ddcba6c918bfc132f",
        "file": "0d82929bf93b8fad1efb114046ae1f913e5c134e09cd939f713cd24d1da8774c",
        "chat": "cb555f4a6519d8b67bc7e1c304b08c0c13852cea297e0fdc644b0b77f70a8f1d",
        "sampled": "5f3620a368decf6e7ad82105879f8ee37cc346961a2332298f700144a3eec488",
        "logits": "939f4dab80703cf2dcfff45656a84cc8fbd6602a8a9c3e6d75ada13ce0c5a5fb",
        "logits_q8_0": "000765286f26f9e922fc23f5dfcbae1b7b53b5574525015b2a956b272252b3f9",
        "logits_q4_0": "5cd1ce3aa760e2917c64c1815e9b32543b4b54059b9eedcb8de09044a3cb61fa",
        "eval": "5083f67f1060e5eb61e61e230f7160683d9fdaf3eb02079b847f202f03419aa7",
    },
    "qwen2.5": {
        "import": "9bf78203fabbc1d93a5f387456ff1d755d7fef36396351c911f51656c898eb99",
        "file": "a7d03b3a04fd4c792738a1474e6bd84513d2f05c5517f1124c461a4254108314",
        "chat": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "sampled": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",  # the same answer
        "logits": "cafb17dbac5ba9251af7a494246a70f258d6de372bab1a273897f07e74993404",
        "logits_q8_0": "e599b1133d04f3c72c0e6e32c24f78c9ea4d67db3424e7e8f0f54c05aa6fa831",
        "logits_q4_0": "67b3cd51a74b369fb4833eaf6392aaf08cd423315df2af9644695e736e3d4925",
        "eval": "30e0fa10d3a5e611d8cf2a7f42e0b13c348054cb9f70130232d553d47f422c59",
    },
    "qwen2.5-1.5b": {
        "import": "626d11f6abd38e28450448eae2574b29de3212825a143f37bdbf145673e8556d",
        "file": "bc8dfb5ea86f37e14ee25033baf06400024d11f2a5532a4a039ea3a7172cd1f9",
        # The same answer tokens as Qwen2.5-0.5B ("The capital of France is Paris."), so the same fingerprint.
        "chat": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "sampled": "7a75f8d55b99478a282d601c155df0fd6c5066b613db230b610612abbdc05c37",
        "logits": "a4eff454791a62e4cfe01c24bd57f9af78809387ae9d699ecf162eeb1007a0a4",
        "logits_q8_0": "fc6ded7859eb616f4baa45870d5e3ad3af9f45757378f288db62d368986b3a73",
        "logits_q4_0": "e4a63bc4b199dfbed9767082df34718a07d3d91230a0209e1161c7bb6138e4a5",
        "eval": "278e17c809204c33ad8d0bf27b655a151638b964918c8e1211ebb9776e7088e9",
    },
    "qwen3": {  # greedy chat rendered with enable_thinking=False
        "import": "c34b4666335e7e0dbe110ee9b9fc78936b84391a57a420688fd78036ad6df5b7",
        "file": "4499bf801c7716d9667ae367c34085bcd2b280264e73a5c7dbca5032e6f9cfc4",
        "chat": "109ee1a8513bf1dad8a34b0483a0f3045fe18cbf7c03820e4e359cc20912201c",
        "sampled": "109ee1a8513bf1dad8a34b0483a0f3045fe18cbf7c03820e4e359cc20912201c",  # the same answer
        "logits": "54400823b346fbacdb126d5ccca19e12f7e8d3c0d04a94ee546abfa840d1523a",
        "logits_q8_0": "21cd894f2d19253192801ec012209624721e1e401bf530b63d39c7e79900eb5a",
        "logits_q4_0": "c49da076a0b5ddb18fcf4984f706eccdc1a044048c334c75b6d021eade247080",
        "eval": "1aaf1cc01dcc873631dd340a7fc40ab9dc4f6ff4281559618a0c23377cfae54f",
    },
    "tinyllama": {
        "import": "b02b4f3085396f3a0b20e07f5b875c9e4a0fbf996dcbc954953bc7d7346b4a4e",
        "file": "5988a45fedbe2602f678679addbfcd9f3dfc19a772bc0afe77d91ac960504cd4",
        "chat": "12608d6f03ed2ea7892990ed181b39a3d0df693637f168ca34008af1403d4bc3",
        "sampled": "12608d6f03ed2ea7892990ed181b39a3d0df693637f168ca34008af1403d4bc3",
        "logits": "1b40cd4dc37c6179feee4abb69e05b42edc3468200e98390685965d893241a1b",
        "logits_q8_0": "5474bab37def485b16798648b8fe338db268e03f1b6544dcb98bc861ff976f53",
        "logits_q4_0": "f0b07f628f930e3f568f480b6bcb0d2c3be737e45495cac2263b03cd38a28f0f",
        "eval": "1767104e39c97f0b1b9c74663c380909b25fbfd2ec49b6ae67e853182b41bd38",
    },
    "olmo2": {
        "import": "183baa6ee6dcc08d855a34f1531b7e5cd1d918c21c6e8bef4cebde7ec675d118",
        "file": "288559f7f1493ae13e14b4e61dda68f53158e4ee19a5170d4bcf674b56e1fc1d",
        "chat": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "sampled": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "logits": "41db902b34618b95d3c185d772cf1f10fa58ee3fbd4094a6fb257252528eb5ba",
        "logits_q8_0": "114b3719f3601bfe781d74af647bb13de3345b6fe9a43b11d49825c6ad461b74",
        "logits_q4_0": "e8488c34ee031d82c7faaaca629aa58ab3766afb3c9b9028a705e6a475249457",
        "eval": "78f5cfddedb0bdd67e799ed89f2cec337eb8e034d6c9284808bdfd61a3f94a58",
    },
    "llama3.2": {
        "import": "1abe366cb3879f73b5fc2c1a6ad191eab99a70b943942bcb98788eff17010f41",
        "file": "8cbc643239199085a3e6a800f0f767385fbf7d0682c63b16c7c3ed826ff7109e",
        # "The capital of France is Paris." in the Llama 3 vocabulary, which OLMo 2 shares: the same fingerprint.
        "chat": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "sampled": "28bbe79bdd0ca41816a367ee7bb30120cb7072991d1eefa2e0cb16a1f1ca8767",
        "logits": "ba7281c7d31afaeb55ed8b259592efd11f71587705584be75ac71197ad02cb73",
        "logits_q8_0": "cea6797804d5b332e889f6d26d7611b31cfa708c296b502a04171198905214e6",
        "logits_q4_0": "44e3f9fc78a682bfddfb99aab425250e8ee49ce7d044ef28f76c8184870042e5",
        "eval": "9ed44b0815c4c8b65a43bbbbb0620b8f2f64155dc61e3b6bf5ddd23aab60eacd",
    },
    "gemma3": {  # the licence text is the Gemma terms link (the repository has no licence file)
        "import": "b03e32d2236a67ff0f06e85c69285df2067ec6951bd3f52c502f13f00915e5e9",
        "file": "ce37a2274fb5048598caee95782b0dd9ef7ba8a9060c2ac26c1da51d7a924c78",
        "chat": "1d3dbbe6c3c890ff1f1215cbd59962f2ced76b538b394b00e1c8d1c97444907f",  # "The capital of France is Paris."
        "sampled": "1d3dbbe6c3c890ff1f1215cbd59962f2ced76b538b394b00e1c8d1c97444907f",  # the same answer
        "logits": "38a0996f57e80c1602e23e89b2897b0cca5422ab49b212e0ee38dfb4886d5360",
        "logits_q8_0": "7546cc1c60e56ed7cc4efd617b81c11ac7f7ebfa8f23e843313be6f850eb864b",
        "logits_q4_0": "0628e1c83908df847416569b6df78cff54766af22258ed98e31a8b7dd83bcf1b",
        "eval": "9d8d8b3c63d6cb7087fbb06f5d101fcdd1a278d0720f3dcea114f6a65410b946",
    },
    # Embedding model: the float32 bits of the EMBEDDING_TEXTS embeddings (last-token pooling, query prompt).
    "qwen3-embedding": {
        "import": "3187399f4e35c2a2f236a8f2ab4435e1fd5f6d9cb40b73ae69c6507fb936066a",
        "embedding": "a2db5e8e1ba53077b201e08ab1616d1dc6681c2c1e283aae5fea5a896f2e6b37",
    },
}

# Trace.fingerprint() of the tiny models over tests/test_interpret.py's PROMPT (every activation, attention included).
TINY_TRACE_FINGERPRINT = {
    "gemma2": "16130094ba98c9b7964034335d783e07f530d6585db8dee7edc52f24db9c2f05",
    "gemma3": "577417f6982891d5594e8ddd516b3c58f01c6f9671e1e00875ad3264c450d401",
    "granite": "e0a5d1d85e59b0c5bce76764395e049a74bd0c496615166e36be998d6c3e8344",
    "llama": "da34729c5b292e870374a74bbece8ef2f3a8373ce49b5f23a1b65c3244edb396",
    "mistral": "8685056d21ed7865b0a41709c6a3619779a4b693d20b27fbe250be557dc00d5a",
    "olmo2": "62dec42fc42006e22c42ea02b756ff9bd7aa668dbcd19d40188dfc0703f69495",
    "phi3": "c6c900b68cd750245988c3f88f0885dd0bb16c3d92b043fc8fb9d87baa02f6b1",
    "qwen2": "9049474d08a5d8987fcbf8ae06f19e0c3f96aa63c7afcccf5343110654cb369c",
    "qwen3": "ae441bb1809c65a97d4b36d0ed01c25cc2454c9439b2da2798c6173309c5cc25",
}

# dllm eval (tests/test_evaluation.py): fingerprints over every per-token log-probability of the bundled tasks
# tests/data/eval-tiny.jsonl (multiple choice) and eval-text.jsonl (perplexity), for the placeholder bigram model and
# the tiny imported Llama of tests/test_engine_import.py.
EVAL_FINGERPRINTS = {
    "bigram_multiple_choice": "5d8c7c3e14555e68584266bf8b330ae362e38152e9eb40adcf27a9f99347700d",
    "bigram_perplexity": "197a7650abe9257cdf2def0085eebbc31ae5540afe20d4ae4b3589e2583332be",
    "tiny_multiple_choice": "95d8333e39566d7f7442630ebdcd695d78906f66ef08c73bdcc827a1740c1e05",
    "tiny_perplexity": "f7c09b64af3a7e5762bf6f8cc6e393f03464c5efb5e873a157b921e964c43e35",
}

# SHA-256 of the conformance vectors' manifest.json (`dllm conformance write`, Phase 20). It covers every input and
# output file's SHA-256, so it pins the bits of every kernel case, the sampler and the two small decoders. Phase 21
# added the two sample-penalties cases (penalties, min-p and logit bias).
CONFORMANCE_MANIFEST_SHA256 = "36b478022962b57a959502bce02235322836f5fc9bea3db927c56cf884800957"

# Phase 21: the placeholder model continuing "Deterministic decoding controls are" with every decoding control set
# (tests/test_decoding_controls.py::CONTROLLED): penalties, min-p and logit bias.
DECODING_CONTROLS_FINGERPRINT = "469d2f4cf54e7b6659fa2b1c34b3a6ae3039be211bb516280f28fc762424b04c"

# Phase 22: SHA-256 of the `dllm batch` output for tests/test_batch_jobs.py's six chat requests and one embedding on
# the placeholder model. The same at every --workers count, on every platform, and after any resume.
BATCH_OUTPUT_SHA256 = "3901563bd7af75cfae606f18f14f0961093583bea2f2963723e778284c5bd934"

# Phase 25. Fingerprint of the model.dllm exported after tests/test_preference.py::RUN (5 DPO steps, beta 0.5, on
# the five preference pairs of that file).
DPO_FINETUNE_FINGERPRINT = {
    "llama": "69d5ef4720169229deed88b5e753d99b2dd145b6fef9be5fc795ae35e8095d7d",
    "qwen3": "06bd4f073a3d8530c1fbbb47002f00e2def08b56f36d158fe0b1b9e1579591ee",
}

# Phase 26. Fingerprints (SHA-256 of [[chunk, score as a hex double], ...]) of tests/test_hybrid_search.py's lexical
# and hybrid rankings of "the capital cat", and of the reranked hybrid passages for "Where is Paris?".
SEARCH_FINGERPRINTS = {
    "lexical": "3b519ea98f501a45a49efaf7e0bfaf4e69275dab26b4d914fb6f9391ca908695",
    "hybrid": "6f385ca748bb98c871344bc3d97cd143bc4abc68db89b41402bf2a153bdc866b",
    "reranked": "b99dea3a6c24f9ca202b15d3115c6b02cd5788cd67d1608605b10e77348cf567",
}

# Phase 27. fingerprint() of tests/test_watermark.py's watermarked generation (tiny model, temperature 0.7, seed 4,
# key "story", delta 6), and the SHA-256 of its detection result as canonical JSON.
WATERMARK_FINGERPRINTS = {
    "generation": "3470b949f942fcbde059fef7380b5a1250992d37fcce73dae0d93ccaf341115a",
    "detection": "b233101d2d72ef04390bcc609427d20aa19459e833b87e2bf48a529fd0a3490d",
}

# Phase 28: the score of a text (test_scoring_voting.TEXT, 3 alternatives) and a vote's receipt output on the tiny
# imported test model.
SCORING_FINGERPRINTS = {
    "score": "4a7a62dd75c5a91cb52b0ef6dc50776b555e54dbe908dd40863e5ae28b3921e1",
    "vote": "d892352c347fc5553d8845f7026be0e6dad98a17afe25ba466b67b75b8476487",
}

GUIDANCE_FINGERPRINTS = {
    "guided": "942e17a71f3f6453980ad236ca1430439fe47d020f9501810f93d8e2133a87a9",
    "contrasted": "d108ec88bfb42113b307508c28f43e5369d6cbaad07bfc10a6e28a09653d79c9",
    "ensembled": "393f94ccc4212df37fcd746b55c35465e6e3e06d1892bb6a5185cc91ddeb591b",
}
"""Fingerprints of guided answers on the test model (tests/test_guidance.py)."""

BEAM_FINGERPRINTS = {
    "chat": "fa459fa24ee3f0362b40a36708a1884cfb6bf91ddc741a430bb842feda5884f0",
    "raw": "4a17e38917e5ee70296ca38876925d7fd2be6a62b3550208ec8d184cb51abe9d",
}
"""Fingerprints of the best beam search answers on the test model (tests/test_beam.py)."""

SCHEMA_FINGERPRINTS = {"person": "f830d3b634b6f3b57323df3a05a72809757dd1393c510ea7f2a14ade7e894aaa"}
"""Fingerprint of an answer constrained by patterns, formats, lengths and bounds (tests/test_schema_constraints.py)."""

GRAMMAR_FINGERPRINTS = {"colours": "6c3e9ca969a5eb689cbf60668385e0136883735a8059c831ddf4328efe2515af"}
"""Fingerprint of an answer constrained by a GBNF grammar (tests/test_gbnf.py)."""

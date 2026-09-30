"""Validation errors and edge cases of the smaller modules: tensors, sampling, tools, the MCP host and friends."""

import json
from dataclasses import dataclass, field

import anyio
import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer
from mcp.types import ImageContent, TextContent

from etalii_dllm.chat import TOOL_CALL_CLOSE, TOOL_CALL_OPEN, ChatMessage, ToolCall
from etalii_dllm.mcp_host import McpHost, McpHostError, McpServerConfig, load_config, parse_server, result_text
from etalii_dllm.models import BigramModel
from etalii_dllm.prompt_cache import PromptCache
from etalii_dllm.sampling import Sampler, SamplingOptions
from etalii_dllm.tensor import Tensor
from etalii_dllm.tokenization import ByteTokenizer
from etalii_dllm.tools import Tool, ToolChoice, parse_calls, template_messages, validate_tools, with_instructions

# -- tensor -----------------------------------------------------------------------------------------------------------


def test_tensor_from_tensor_shares_storage():
    tensor = Tensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    again = Tensor(tensor)
    assert again.numpy() is tensor.numpy()
    assert again == tensor and not again.numpy().flags.writeable


def test_tensor_properties_and_repr():
    tensor = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    assert (tensor.ndim, tensor.size, len(tensor), tensor.strides) == (3, 24, 2, (12, 4, 1))
    assert repr(tensor) == "Tensor(shape=(2, 3, 4))"
    assert len(Tensor.zeros([5, 1])) == 5


def test_tensor_array_protocol_converts_and_copies_on_request():
    tensor = Tensor([1.5, -2.0, 3.25])
    as_double = np.asarray(tensor, dtype=np.float64)
    assert as_double.dtype == np.float64 and as_double.tolist() == [1.5, -2.0, 3.25]
    assert np.asarray(tensor) is tensor.numpy()
    copied = np.array(tensor, copy=True)
    assert copied is not tensor.numpy() and copied.flags.writeable
    copied[0] = 0  # the tensor itself stays untouched
    assert tensor.numpy()[0] == 1.5


def test_tensor_equality_is_bitwise_and_only_with_tensors():
    tensor = Tensor([0.0, 1.0])
    assert tensor != [0.0, 1.0]
    assert tensor.__eq__(np.array([0.0, 1.0], dtype=np.float32)) is NotImplemented
    assert tensor != Tensor([-0.0, 1.0])  # equal numbers, different bits
    assert tensor != Tensor([[0.0, 1.0]])  # same bits, different shape
    nan = Tensor([np.nan])
    assert nan == Tensor([np.nan])


# -- sampling ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"temperature": -0.1}, "temperature must be non-negative"),
        ({"top_p": 0.0}, r"top_p must be in \(0, 1\]"),
        ({"top_p": 1.5}, r"top_p must be in \(0, 1\]"),
        ({"top_k": -1}, "top_k must be non-negative"),
    ],
)
def test_sampling_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        SamplingOptions(**options)


def test_top_p_keeps_the_smallest_nucleus():
    logits = np.log(np.array([0.05, 0.6, 0.3, 0.05], dtype=np.float32))
    # 0.6 alone reaches top_p 0.5: every draw is token 1, whatever the seed.
    assert {Sampler(SamplingOptions(temperature=1.0, top_p=0.5, seed=s)).sample(logits) for s in range(50)} == {1}
    # 0.6 + 0.3 reaches 0.85: tokens 1 and 2 only, and both occur.
    drawn = {Sampler(SamplingOptions(temperature=1.0, top_p=0.85, seed=s)).sample(logits) for s in range(200)}
    assert drawn == {1, 2}


def test_top_p_ties_are_broken_by_token_id():
    logits = np.zeros(4, dtype=np.float32)
    # Four equal tokens: the nucleus of 0.5 is the two lowest ids.
    drawn = {Sampler(SamplingOptions(temperature=1.0, top_p=0.5, seed=s)).sample(logits) for s in range(200)}
    assert drawn == {0, 1}


# -- chat, models, tokenization, prompt cache -------------------------------------------------------------------------


def test_tool_call_with_invalid_json_arguments_keeps_the_text():
    call = ToolCall("call_1", "lookup", "{not json")
    assert call.arguments_object() == "{not json"
    assert call.render() == f'{TOOL_CALL_OPEN}\n{{"name": "lookup", "arguments": "{{not json"}}\n{TOOL_CALL_CLOSE}'


def test_bigram_model_needs_a_vocabulary():
    with pytest.raises(ValueError, match="vocabulary_size must be positive"):
        BigramModel(0, seed=1)


def test_byte_tokenizer_decode_drops_special_tokens_and_replaces_broken_utf8():
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode("hé")
    assert tokenizer.decode([*tokens, tokenizer.end_of_sequence]) == "hé"
    assert tokenizer.decode(tokens[:-1]) == "h�"


@dataclass
class FakeCache:
    tokens: list[int] = field(default_factory=list)


def test_prompt_cache_capacity_and_clear():
    with pytest.raises(ValueError, match="the prompt cache capacity must be non-negative"):
        PromptCache(FakeCache, capacity=-1)
    pool = PromptCache(FakeCache, capacity=2)
    pool.release(FakeCache([1, 2, 3]))
    assert len(pool) == 1
    pool.clear()
    assert len(pool) == 0
    cache, saved = pool.acquire([1, 2, 3, 4])
    assert (cache.tokens, saved) == ([], 0)


# -- tools ------------------------------------------------------------------------------------------------------------

LOOKUP = Tool("lookup", "Looks a word up", {"type": "object", "properties": {"q": {"type": "string"}}})


def test_tool_choice_and_names_are_validated():
    with pytest.raises(ValueError, match="unknown tool_choice 'any'"):
        ToolChoice("any")
    for name in ("has space", "", "x" * 129, "café"):
        with pytest.raises(ValueError, match="invalid tool name"):
            validate_tools([Tool(name)], ToolChoice())
    validate_tools([Tool("ns.tool_1:v-2")], ToolChoice())


def test_hermes_conversion_merges_tool_results_and_extends_the_system_message():
    call = ToolCall("call_1", "lookup", '{"q": "a"}')
    messages = [
        ChatMessage("system", "Be brief."),
        ChatMessage("user", "Look up a and b"),
        ChatMessage("assistant", "", tool_calls=(call,)),
        ChatMessage("tool", "first", tool_call_id="call_1", name="lookup"),
        ChatMessage("tool", "second", tool_call_id="call_2", name="lookup"),
    ]
    converted = with_instructions(messages, [LOOKUP])
    assert [m.role for m in converted] == ["system", "user", "assistant", "user"]
    assert converted[0].content.startswith("Be brief.\n\n# Tools\n")
    assert '"name": "lookup"' in converted[0].content
    assert converted[3].content == (
        "<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response>"
    )
    # Without tools the conversation is only converted, no instructions are added.
    assert with_instructions(messages[:2], []) == messages[:2]


def test_template_messages_carry_tool_call_ids_and_names():
    call = ToolCall("call_1", "lookup", '{"q": "a"}')
    entries = template_messages(
        [
            ChatMessage("assistant", "", tool_calls=(call,)),
            ChatMessage("tool", "found", tool_call_id="call_1", name="lookup"),
        ]
    )
    assert entries == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": {"q": "a"}}}
            ],
        },
        {"role": "tool", "content": "found", "tool_call_id": "call_1", "name": "lookup"},
    ]


@pytest.mark.parametrize(
    "block",
    [
        "not json at all",
        '{"name": "lookup", "arguments": "{broken"}',
        '{"name": "lookup", "arguments": [1, 2]}',
        '{"name": "lookup", "arguments": "[1, 2]"}',
        '["lookup"]',
        '{"name": "unknown", "arguments": {}}',
    ],
)
def test_blocks_that_are_not_calls_stay_in_the_text(block):
    text = f"Sure. {TOOL_CALL_OPEN}{block}{TOOL_CALL_CLOSE}"
    assert parse_calls(text, [LOOKUP]) == (text, [])


def test_string_and_parameters_arguments_are_accepted():
    text = f'{TOOL_CALL_OPEN}{{"name": "lookup", "arguments": "{{\\"q\\": \\"a\\"}}"}}{TOOL_CALL_CLOSE}'
    assert parse_calls(text, [LOOKUP]) == ("", [("lookup", '{"q": "a"}')])
    assert parse_calls('{"name": "lookup", "parameters": {"q": "b"}}', [LOOKUP]) == ("", [("lookup", '{"q": "b"}')])
    assert parse_calls("{not a call", [LOOKUP]) == ("{not a call", [])


# -- MCP host ---------------------------------------------------------------------------------------------------------


def test_mcp_configuration_errors_name_the_problem():
    with pytest.raises(McpHostError, match="MCP server 'broken' must be an object"):
        load_config({"mcpServers": {"broken": "uvx server"}})
    with pytest.raises(McpHostError, match="empty MCP server specification 'name='"):
        parse_server("name=")
    with pytest.raises(McpHostError, match="empty MCP server specification '  '"):
        parse_server("  ")
    with pytest.raises(McpHostError, match="MCP server names must be unique"):
        McpHost([McpServerConfig("a", command="x"), McpServerConfig("a", url="http://localhost/mcp")])


def test_result_text_renders_non_text_blocks_and_structured_content():
    image = ImageContent(type="image", data="aGk=", mimeType="image/png")
    text = result_text([TextContent(type="text", text="caption"), image])
    first, second = text.split("\n")
    assert first == "caption"
    assert json.loads(second) == {"type": "image", "data": "aGk=", "mimeType": "image/png"}
    assert second == json.dumps(json.loads(second), sort_keys=True)
    assert result_text([], {"b": 1, "a": "é"}) == '{"a": "é", "b": 1}'
    assert result_text([TextContent(type="text", text="only")], {"ignored": True}) == "only"
    assert result_text([]) == ""


def test_unreachable_url_server_is_named():
    async def go():
        async with McpHost([McpServerConfig("web", url="http://127.0.0.1:9/mcp")]):
            pass

    with pytest.raises(McpHostError, match="cannot connect to MCP server 'web'"):
        anyio.run(go)


def test_call_on_a_closed_host_becomes_an_error_result():
    server = MCPServer("fixture")

    @server.tool()
    def add(a: int, b: int) -> str:
        """Adds two integers."""
        return str(a + b)

    async def go():
        async with McpHost({"fixture": server}) as host:
            ok = await host.call(ToolCall("call_1", "add", '{"a": 2, "b": 3}'))
        return ok, await host.call(ToolCall("call_2", "add", '{"a": 2, "b": 3}'))

    ok, closed = anyio.run(go)
    assert (ok.content, ok.is_error) == ("5", False)
    assert closed.is_error and closed.server == "fixture"
    assert closed.content.startswith("RuntimeError: ")

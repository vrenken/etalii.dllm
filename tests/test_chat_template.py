"""Chat template rendering with the Hugging Face ``apply_chat_template`` environment."""

from __future__ import annotations

import datetime

import pytest

from etalii_dllm.chat_template import ChatTemplate, ChatTemplateError

# SmolLM2-Instruct's template (ChatML with a default system prompt).
SMOLLM2 = (
    "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
    "{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}"
    "{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

MESSAGES = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "2+2?"},
]


def test_chatml_rendering():
    prompt = ChatTemplate(SMOLLM2).render(MESSAGES)
    assert prompt == (
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n"
        "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert not ChatTemplate(SMOLLM2).render(MESSAGES, add_generation_prompt=False).endswith("assistant\n")


def test_block_trimming_special_tokens_and_tools():
    template = ChatTemplate(
        "{{ bos_token }}\n{% for m in messages %}\n  [{{ m.role }}] {{ m.content }}\n{% endfor %}\n"
        "{% if tools %}{{ tools | tojson }}{% endif %}",
        special_tokens={"bos_token": "<s>", "eos_token": None},
    )
    tools = [{"type": "function", "function": {"name": "météo", "parameters": {}}}]
    assert template.render(MESSAGES[:1], tools=tools) == (
        '<s>\n  [user] Hi\n[{"type": "function", "function": {"name": "météo", "parameters": {}}}]'
    )


def test_raise_exception_and_errors():
    template = ChatTemplate("{% if messages[0]['role'] != 'system' %}{{ raise_exception('system first') }}{% endif %}")
    with pytest.raises(ChatTemplateError, match="system first"):
        template.render(MESSAGES)
    with pytest.raises(ChatTemplateError):
        ChatTemplate("{% for %}")


def test_date_comes_from_the_caller_not_the_clock():
    template = ChatTemplate("Today: {{ strftime_now('%d %b %Y') }}")
    assert template.render([], date=datetime.date(2026, 9, 28)) == "Today: 28 Sep 2026"
    with pytest.raises(ChatTemplateError, match="date"):
        template.render([])


def test_sandbox_blocks_attribute_escapes():
    template = ChatTemplate("{{ messages.__class__.__mro__ }}")
    with pytest.raises(ChatTemplateError):
        template.render(MESSAGES)

import json

from lmconf.config import OpenaiCompatibleLLMConf
from pytest_httpx import HTTPXMock


def test_llmconf_openai_create_lc_chat_model(httpx_mock: HTTPXMock):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import AIMessage

    openai_llmconf = OpenaiCompatibleLLMConf(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key="sk-1234",
        base_url="https://api.openai.com/v1",
    )
    chat_model = openai_llmconf.create_langchain_chatmodel()
    assert isinstance(chat_model, ChatOpenAI)

    resp_json = """\
{
    "choices": [{"finish_reason": "stop",
                 "index": 0,
                 "message": {
                    "content": "Hello! How can I assist you today?",
                    "role": "assistant"
                }}],
    "created": 1721397940,
    "id": "chatcmpl-9miWqoPcY",
    "model": "gpt-3.5-turbo",
    "object": "chat.completion",
    "system_fingerprint": null,
    "usage": {"completion_tokens": 9,
              "prompt_tokens": 8,
              "total_tokens": 17}
}"""
    httpx_mock.add_response(json=json.loads(resp_json))
    message = chat_model.invoke("Hello!")
    assert isinstance(message, AIMessage)
    assert message.content == "Hello! How can I assist you today?"

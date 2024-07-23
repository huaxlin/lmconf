import os
import uuid
from tempfile import TemporaryDirectory
from textwrap import dedent

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print

from lmconf import LMConfSettings
from lmconf.llm_configs.tongyi import TongyiLLMConf


class Settings(BaseSettings, LMConfSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="lmconf_",
        env_nested_delimiter="__",
    )


@pytest.fixture
def settings():
    dotenv_content = dedent(
        """\
        lmconf_lm_config__config_list='[
            {"name": "local",
             "conf": {"provider": "ollama",
                      "model": "tinyllama",
                      "base_url": "http://localhost:11434"}},
            {"name": "azure_us",
             "conf": {"provider": "azure_openai",
                      "model": "gpt-35-turbo",
                      "api_version": "2023-05-15",
                      "base_url": "https://${COMPANY}-gpt.openai.azure.com",
                      "api_key": "${AZURE_US_API_KEY}"}},
            {"name": "azure_je",
             "conf": {"provider": "azure_openai",
                      "model": "gpt-35-turbo",
                      "api_version": "2023-05-15",
                      "base_url": "https://${COMPANY}-gpt-je.openai.azure.com",
                      "api_key": "${AZURE_JE_API_KEY}"}},
            {"name": "zh_llm",
             "conf": {"provider": "tongyi",
                       "model": "qwen-max",
                       "api_key": "${DASHSCOPE_API_KEY}"}}
        ]'
        lmconf_lm_config__x='{
            "chatbot": ["local"],
            "rag": ["azure_us", "gpt-35-turbo"],
            "tool-use": ["azure_je", "gpt-4"],
            "ReAct": ["zh_llm","qwen-turbo"]
        }'
        """
    )
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ".env"), "w") as f:
            f.write(dotenv_content)
        _cwd = os.getcwd()
        os.chdir(tmpdir)
        settings = Settings()
        os.chdir(_cwd)
    return settings


def test_local_ollama(settings: Settings):
    llm = settings.lm_config.get("chatbot").create_langchain_chatmodel(temperature=0.1)
    messages: list[BaseMessage] = [
        HumanMessage(content="I'm Bob. What is the capital of France?")
    ]
    output1 = llm.invoke(messages)
    print(output1)
    assert isinstance(output1, AIMessage) and isinstance(output1.content, str)
    assert "Paris" in output1.content

    messages.append(output1)
    messages.append(HumanMessage(content="What's my name?"))
    output2 = llm.invoke(messages)
    print(output2)
    assert isinstance(output2, AIMessage) and isinstance(output2.content, str)
    assert "Bob" in output2.content


def test_tongyi(settings: Settings):
    ai_functionality_name = "ReAct"
    # LLM provider and model designated for AI functionality
    named_config, which_model = settings.lm_config.x[ai_functionality_name]
    assert named_config == "zh_llm"
    assert which_model == "qwen-turbo"
    # find LLM Configurations by AI funtion name
    finding = filter(
        lambda d: d["name"] == named_config, settings.lm_config.config_list
    )
    found_llm_conf = list(finding)[0]["conf"]
    assert isinstance(found_llm_conf, TongyiLLMConf)
    assert found_llm_conf.provider == "tongyi"
    assert found_llm_conf.model == "qwen-max"

    # default dashscope package HTTP API
    import dashscope

    assert os.environ.get("DASHSCOPE_HTTP_BASE_URL", None) is None
    assert dashscope.base_http_api_url == "https://dashscope.aliyuncs.com/api/v1"

    llm = settings.lm_config.get(ai_functionality_name).create_langchain_chatmodel()
    output1 = llm.invoke("你是谁？")
    print(output1)
    assert isinstance(output1, AIMessage) and isinstance(output1.content, str)
    # Use AI function specified model instead of `lm_config` configured model
    assert output1.response_metadata["model_name"] == which_model
    assert "通义千问" in output1.content


def test_azure_openai(settings: Settings):
    from langchain_openai import AzureChatOpenAI

    azure_llm = settings.lm_config.get("rag").create_langchain_chatmodel(
        temperature=0.1
    )
    assert isinstance(azure_llm, AzureChatOpenAI)

    output1 = azure_llm.invoke("Hello")
    print(output1)
    assert isinstance(output1, AIMessage) and isinstance(output1.content, str)
    assert output1.content == "Hello! How can I assist you today?"

    azure_model = settings.lm_config.get("tool-use").create_langchain_chatmodel()
    assert isinstance(azure_model, AzureChatOpenAI)


@pytest.mark.reverse_proxy
def test_tongyi_proxy_base_url():
    from langchain_community.chat_models.tongyi import ChatTongyi

    settings = Settings.model_validate(
        {
            "lm_config": {
                "config_list": [
                    {
                        "name": "proxy_dashscope",
                        "conf": {
                            "provider": "tongyi",
                            "model": "qwen-turbo",
                            "api_key": "${DASHSCOPE_API_KEY}",
                            "base_url": "http://localhost:6880/api/v1",
                        },
                    }
                ],
                "x": {"proxy-tongyi": ["proxy_dashscope"]},
            },
        }
    )

    llm = settings.lm_config.get("proxy-tongyi").create_langchain_chatmodel()
    assert isinstance(llm, ChatTongyi)

    # overrode dashscope package HTTP API
    import dashscope

    assert os.environ.get("DASHSCOPE_HTTP_BASE_URL", None) is None
    assert dashscope.base_http_api_url == "http://localhost:6880/api/v1"

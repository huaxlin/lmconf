import os
import sys
from pathlib import Path

import pytest
from rich import print


@pytest.fixture
def examples_into_syspath(examples_dir: Path):
    try:
        yield sys.path.insert(0, str(examples_dir))
    finally:
        sys.path.remove(str(examples_dir))


def test_chatbot(examples_into_syspath):
    import uuid
    from tempfile import TemporaryDirectory
    from textwrap import dedent

    from chatbot import Chatbot, get_current_settings
    from langchain_core.messages import AIMessage

    # fmt: off
    dotenv_content = dedent("""\
        lc_chatbot_SQLALCHEMY_DATABASE_URI=sqlite:///db.sqlite3
        lc_chatbot_lm_config__config_list='[
            {"name": "local",
             "conf": {"provider": "ollama",
                      "model": "gemma:2b",
                      "base_url": "http://localhost:11434"}},
            {"name": "azure_us",
             "conf": {"provider": "azure_openai",
                      "model": "gpt-35-turbo",
                      "base_url": "https://${COMPANY}-gpt.openai.azure.com",
                      "api_key": "${AZURE_US_API_KEY}"}},
            {"name": "azure_je",
             "conf": {"provider": "azure_openai",
                        "model": "gpt-35-turbo",
                        "base_url": "https://${COMPANY}-gpt-je.openai.azure.com",
                        "api_key": "${AZURE_JE_API_KEY}"}},
            {"name": "zh_llm",
             "conf": {"provider": "tongyi",
                      "model": "qwen-turbo",
                      "api_key": "${DASHSCOPE_API_KEY}"}}
        ]'
        lc_chatbot_lm_config__x='{
            "chatbot": ["local"]
        }'
    """)
    # fmt: on
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ".env"), "w") as f:
            f.write(dotenv_content)
        _cwd = os.getcwd()
        os.chdir(tmpdir)
        settings = get_current_settings()  # make cache
        os.chdir(_cwd)

    print(get_current_settings())
    session_id = uuid.uuid4().hex
    chatbot = Chatbot(session_id)

    msg1 = "Hi! I'm bob"
    output1 = chatbot(msg1)
    print(output1)
    assert isinstance(output1, AIMessage)
    assert isinstance(output1.content, str)

    msg2 = "Whats my name"
    output2 = chatbot(msg2)
    print(output2)
    assert isinstance(output2, AIMessage)
    assert isinstance(output2.content, str)

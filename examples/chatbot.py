from __future__ import annotations

import os
from functools import partial
from typing import Optional

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic_settings import BaseSettings, SettingsConfigDict

from lmconf import LMConfSettings


class Settings(BaseSettings, LMConfSettings):
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="lc_chatbot_",
        env_nested_delimiter="__",
    )


_FROM_ENV_CACHE: dict[int, Settings] = {}


def get_current_settings() -> Settings:
    cache_key = hash(tuple((key, value) for key, value in os.environ.items()))

    if cache_key not in _FROM_ENV_CACHE:
        _FROM_ENV_CACHE[cache_key] = Settings()

    return _FROM_ENV_CACHE[cache_key]


class Chatbot:
    def __init__(self, session_id: str, connection: str | None = None):
        settings = get_current_settings()
        connection = connection or settings.SQLALCHEMY_DATABASE_URI

        self.session_id = session_id
        self.connection = connection

        llm = settings.lm_config.get("chatbot").create_langchain_chatmodel(
            temperature=0.1
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm

        _SQLChatMessageHistory = partial(SQLChatMessageHistory, connection=connection)
        self.chatbot = RunnableWithMessageHistory(
            chain,
            lambda session_id: _SQLChatMessageHistory(session_id=session_id),
            input_messages_key="question",
            history_messages_key="history",
        )
        self.config = {"configurable": {"session_id": session_id}}

    def __call__(self, question):
        message = self.chatbot.invoke({"question": question}, config=self.config)
        return message


if __name__ == "__main__":
    import uuid
    from tempfile import TemporaryDirectory
    from textwrap import dedent

    from rich import print

    # fmt: off
    dotenv_content = dedent("""\
        lc_chatbot_SQLALCHEMY_DATABASE_URI=sqlite:///db.sqlite3
        lc_chatbot_lm_config__config_list='{
            "local": {"provider": "ollama",
                    "model": "gemma:2b",
                    "base_url": "http://localhost:11434"},
            "azure_us": {"provider": "azure_openai",
                        "model": "gpt-35-turbo",
                        "base_url": "https://${COMPANY}-gpt.openai.azure.com",
                        "api_key": "${AZURE_US_API_KEY}"},
            "azure_je": {"provider": "azure_openai",
                        "model": "gpt-35-turbo",
                        "base_url": "https://${COMPANY}-gpt-je.openai.azure.com",
                        "api_key": "${AZURE_JE_API_KEY}"},
            "zh_llm": {"provider": "tongyi",
                    "model": "qwen-turbo",
                    "api_key": "${DASHSCOPE_API_KEY}"}
        }'
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
    print(f"{session_id = }")

    chatbot = Chatbot(session_id)

    message = chatbot("Hi! I'm bob")
    print(message)

    message = chatbot("Whats my name")
    print(message)

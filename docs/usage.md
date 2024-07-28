# lmconf Usage

This document provides a comprehensive guide on utilizing lmconf for configuring and managing large language models (LLMs) in your Python applications.

## Installation

Install lmconf using pip:

```bash
pip install lmconf
```

You can also install optional dependencies for specific LLM providers:

- LangChain integrations:

  ```bash
  pip install lmconf[langchain]
  ```

- Tongyi (Alibaba Cloud):

  ```bash
  pip install lmconf[tongyi]
  ```

## Basic Configuration
lmconf leverages environment variables and a structured configuration to define and manage LLM configurations. You can define your LLM configurations within a `.env` file or directly within your code.

### Using a `.env` file

Create a `.env` file in your project root and configure your LLMs as follows:

```
# Example .env file
LMCONF_lm_config__config_list='[
    {"name": "azure_us",
     "conf": {
        "provider": "azure_openai",
        "model": "gpt-35-turbo",
        "api_version": "2023-05-15",
        "base_url": "https://${COMPANY}-gpt.openai.azure.com",
        "api_key": "${AZURE_US_API_KEY}"
     }},
    {"name": "zh_llm",
     "conf": {
        "provider": "tongyi",
        "model": "qwen-max",
        "api_key": "${DASHSCOPE_API_KEY}"
     }}
]'

LMCONF_lm_config__x='{
    "chatbot": ["azure_us", "gpt-35-turbo"],
    "rag": ["zh_llm", "qwen-turbo"]
}'
```

Explanation:

- `LMCONF_lm_config__config_list`: Defines a list of LLM configurations. Each configuration is a dictionary with:
  - name: A unique identifier for the configuration.
  - conf: A dictionary containing LLM-specific details like provider, model, API key, and base URL.
- `LMCONF_lm_config__x`: Maps functionalities (like "chatbot" or "rag") to LLM configurations. This allows you to quickly select the appropriate LLM for a specific purpose.


## Using code

You can also define your LLM configurations directly within your Python code:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from lmconf import LMConfSettings

class Settings(BaseSettings, LMConfSettings):
    # ... other settings ...

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LMCONF_",
        env_nested_delimiter="__",
    )

settings = Settings()
```

### Accessing LLMs
Once you have defined your configurations, you can use lmconf to retrieve and use the LLMs:

```python
# Example using Settings class
chatbot_llm = settings.lm_config.get("chatbot").create_langchain_chatmodel(temperature=0.1)
output = chatbot_llm.invoke("What is the capital of France?")
print(output)

# Example using LMConfig directly
lm_config = LMConfig(
    config_list=[
        {
            "name": "azure_us",
            "conf": {
                "provider": "azure_openai",
                "model": "gpt-35-turbo",
                "api_key": "your-api-key",
                "base_url": "https://your-company-gpt.openai.azure.com",
                "api_version": "2023-05-15",
            }
        }
    ],
    x={
        "chatbot": ["azure_us", "gpt-35-turbo"],
    }
)

chatbot_llm = lm_config.get("chatbot").create_langchain_chatmodel(temperature=0.1)
output = chatbot_llm.invoke("What is the capital of France?")
print(output)
```

## Example: Using lmconf with LangChain

This example demonstrates how to use lmconf with LangChain for creating a simple chatbot:

```python
from lmconf import LMConfSettings
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings, LMConfSettings):
    SQLALCHEMY_DATABASE_URI: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="lc_chatbot_",
        env_nested_delimiter="__",
    )

settings = Settings()

class Chatbot:
    def __init__(self, session_id: str):
        self.session_id = session_id

        llm = settings.lm_config.get("chatbot").create_langchain_chatmodel(temperature=0.1)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        chain = prompt | llm

        self.chatbot = RunnableWithMessageHistory(
            chain,
            lambda session_id: SQLChatMessageHistory(session_id=session_id, connection=settings.SQLALCHEMY_DATABASE_URI),
            input_messages_key="question",
            history_messages_key="history",
        )

    def __call__(self, question):
        message = self.chatbot.invoke({"question": question}, config={"configurable": {"session_id": self.session_id}})
        return message

if __name__ == "__main__":
    import uuid
    chatbot = Chatbot(uuid.uuid4().hex)

    message = chatbot("Hello! I'm Bob.")
    print(message)

    message = chatbot("What's my name?")
    print(message)
```

## Advanced Features

### Custom LLM Configurations

You can create custom LLM configuration classes by extending `LLMConfBase` from lmconf. This allows you to define provider-specific parameters and implement custom logic for creating LangChain models.

### Environmental Variables

lmconf utilizes environment variables for flexible configuration. You can set environment variables in your `.env` file or directly using shell commands.

### Caching

`EnvCacheSettingsMixin` uses a caching mechanism to avoid unnecessary validation and improve performance when accessing LLM configurations from environment variables.

### Conclusion

lmconf simplifies the configuration and management of LLMs in your Python applications, providing a robust and flexible framework for integrating LLMs into your projects.

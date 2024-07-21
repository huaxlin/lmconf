from typing_extensions import Literal

from lmconf.config import OpenAICompatibleLLMConf


class AzureOpenAILLMConf(OpenAICompatibleLLMConf):
    provider: Literal["azure_openai"] = "azure_openai"
    api_version: str

    def create_langchain_chatmodel(self, **chatmodel_kwargs):
        import langchain_openai

        return langchain_openai.AzureChatOpenAI(
            azure_endpoint=self.base_url,
            deployment_name=self.model,
            openai_api_version=self.api_version,
            openai_api_key=self.api_key,
            **chatmodel_kwargs,
        )

    def create_langchain_llm(self, **llm_kwargs):
        import langchain_openai

        return langchain_openai.AzureOpenAI(
            azure_endpoint=self.base_url,
            deployment_name=self.model,
            openai_api_version=self.api_version,
            openai_api_key=self.api_key,
            **llm_kwargs,
        )

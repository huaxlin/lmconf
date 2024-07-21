from typing_extensions import Literal
from pydantic import field_validator
from lmconf.config import OpenAICompatibleLLMConf


class TongyiLLMConf(OpenAICompatibleLLMConf):
    provider: Literal["tongyi"] = "tongyi"
    model: str = "qwen-turbo"

    @field_validator("base_url", mode="after")
    @classmethod
    def set_dashscope_base_url(cls, v):
        if v:
            import dashscope

            dashscope.base_http_api_url = v
        return v

    def create_langchain_chatmodel(self, **chatmodel_kwargs):
        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(
            model_name=self.model,
            dashscope_api_key=self.api_key,
            **chatmodel_kwargs,
        )

    def create_langchain_llm(self, **llm_kwargs):
        from langchain_community.llms.tongyi import Tongyi

        return Tongyi(
            model_name=self.model,
            dashscope_api_key=self.api_key,
            **llm_kwargs,
        )

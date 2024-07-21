from typing_extensions import Literal
from pydantic import model_validator
from lmconf.config import OpenAICompatibleLLMConf


class TongyiLLMConf(OpenAICompatibleLLMConf):
    provider: Literal["tongyi"] = "tongyi"
    model: str = "qwen-turbo"

    @model_validator(mode="after")
    def set_dashscope_base_url(self):
        # TODO: set base_http_api_url when dashscope.Generation.call is called
        if self.base_url:
            import dashscope

            dashscope.base_http_api_url = self.base_url
        return self

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

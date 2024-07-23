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

    def create_langchain_chatmodel(self, **lc_kwargs):
        from langchain_community.chat_models.tongyi import ChatTongyi

        if "temperature" in lc_kwargs:
            model_kwargs = lc_kwargs.pop("model_kwargs", {})
            model_kwargs["temperature"] = lc_kwargs.pop("temperature")
            lc_kwargs["model_kwargs"] = model_kwargs

        return ChatTongyi(
            model_name=self.model,
            dashscope_api_key=self.api_key,
            **lc_kwargs,
        )

    def create_langchain_llm(self, **lc_kwargs):
        from langchain_community.llms.tongyi import Tongyi

        if "temperature" in lc_kwargs:
            model_kwargs = lc_kwargs.pop("model_kwargs", {})
            model_kwargs["temperature"] = lc_kwargs.pop("temperature")
            lc_kwargs["model_kwargs"] = model_kwargs

        return Tongyi(
            model_name=self.model,
            dashscope_api_key=self.api_key,
            **lc_kwargs,
        )

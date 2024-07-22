from typing import Optional
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.language_models.llms import BaseLLM


class LLMConfBase(BaseModel):
    provider: str = Field(description='e.g. "ollama", "openai", "tongyi", "azure_openai"')
    model: str = Field(description="default model if not set")

    def create_langchain_chatmodel(self, *args, **kwargs) -> "BaseChatModel":
        raise NotImplementedError()

    def create_langchain_llm(self, *args, **kwargs) -> "BaseLLM":
        raise NotImplementedError()


class OpenAICompatibleLLMConf(LLMConfBase):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def create_langchain_chatmodel(self, **chatmodel_kwargs):
        try:
            import langchain_community  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Could not import langchain_community python package. "
                "Please install it with `pip install lmconf[langchain]`."
            ) from exc
        try:
            import langchain_openai
        except ImportError as exc:
            raise ImportError(
                "Could not import langchain_openai python package. "
                "Please install it with `pip install lmconf[langchain]`."
            ) from exc
        from langchain_community import chat_models as lccm

        mapping_cls_name = {
            _module.split(".")[-1]: cls_name
            for cls_name, _module in lccm._module_lookup.items()
        }
        if self.provider == "openai":
            chat_model_cls = langchain_openai.ChatOpenAI
        elif self.provider == "azure_openai":
            chat_model_cls = langchain_openai.AzureChatOpenAI
        elif self.provider in mapping_cls_name:
            chat_model_cls = getattr(lccm, mapping_cls_name[self.provider])
        else:
            raise ValueError(
                f"Unsupported convert to LangChain ChatModel with provider: {self.provider}"
            )
        return chat_model_cls(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            **chatmodel_kwargs,
        )

    def create_langchain_llm(self, **llm_kwargs):
        try:
            import langchain_community  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Could not import langchain_community python package. "
                "Please install it with `pip install lmconf[langchain]`."
            ) from exc
        try:
            import langchain_openai
        except ImportError as exc:
            raise ImportError(
                "Could not import langchain_openai python package. "
                "Please install it with `pip install lmconf[langchain]`."
            ) from exc

        from langchain_community.llms import get_type_to_cls_dict

        lc_provider_to_cls_getter_mapper = get_type_to_cls_dict()

        if self.provider == "openai":
            llm_cls = langchain_openai.OpenAI
        elif self.provider == "azure_openai":
            llm_cls = langchain_openai.AzureOpenAI
        elif self.provider in lc_provider_to_cls_getter_mapper:
            llm_cls = lc_provider_to_cls_getter_mapper[self.provider]()
        else:
            raise ValueError(
                f"Unsupported convert to LangChain LLM with provider: {self.provider}"
            )
        return llm_cls(
            model=self.model, api_key=self.api_key, base_url=self.base_url, **llm_kwargs
        )

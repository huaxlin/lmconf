from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class LLMConfBase(BaseModel):
    provider: str = Field(description='e.g. "openai", "tongyi", "azure_openai"')
    model: str = Field(description="default model if not set")


class OpenaiCompatibleLLMConf(LLMConfBase):
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def as_langchain_chatmodel(self):
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
        # TODO: inherit ChatTongyi to support override base_url
        # from langchain_community.chat_models.tongyi import ChatTongyi
        return chat_model_cls(
            model=self.model, api_key=self.api_key, base_url=self.base_url
        )

    def as_langchain_llm(self):
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
        return llm_cls(model=self.model, api_key=self.api_key, base_url=self.base_url)


class LMConfig(BaseModel):
    """
    ```json
    {
        "lm_config.x": {"foo": ["azure_je"], // will use default model: gpt-35-turbo
                        "bar": ["azure_us","gpt-4"],  // use specific model: gpt-4
                        "egg": ["zh_llm"]},  // will use default model: qwen-turbo
        "lm_config.config_list": {
          "azure_je": {"provider": "azure_openai", "model": "gpt-35-turbo", ...},
          "azure_us": {"provider": "azure_openai", "model": "gpt-35-turbo", ...},
          "zh_llm": {"provider": "tongyi", "model": "qwen-turbo", ...},
        }
    }
    ```
    """

    x: dict = Field(default_factory=dict)
    # named-provider -> llm-config(provider, model, api_key, base_url, ...)
    config_list: Dict[str, OpenaiCompatibleLLMConf] = Field(default_factory=dict)

    def get(
        self,
        name: Optional[str] = None,
        named_provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Union[LLMConfBase, None]:
        """
        Args:
            name (str): the named of function which uses the specific LLM from the config.
            named_provider (str): if not use named of function, use this named of the provider to get the LLMConf object.

        Returns:
            the LLMConf object specified by the name.
        """
        if not name and not named_provider:
            raise ValueError("name or named_provider must be specified")
        if name:
            if name not in self.x:
                raise ValueError(f"{name} not found in lm_config.x")
            which_llm: list = self.x[name]
            named_provider, model = (
                (which_llm[0], None) if len(which_llm) < 2 else which_llm
            )

        default_llm_conf = self.config_list[named_provider]  # type: ignore
        if not model:
            return default_llm_conf
        return default_llm_conf.model_copy(update={"model": model})


class LMConfSettings:
    lm_config: LMConfig = Field(default_factory=LMConfig)

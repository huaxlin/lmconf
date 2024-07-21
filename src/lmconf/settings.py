from typing import Dict, Optional

from pydantic import BaseModel, Field

from lmconf.config import LLMConfBase, OpenaiCompatibleLLMConf


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
    #  TODO: build LMConf instance by provider
    config_list: Dict[str, OpenaiCompatibleLLMConf] = Field(default_factory=dict)

    def get(
        self,
        name: Optional[str] = None,
        named_provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> LLMConfBase:
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

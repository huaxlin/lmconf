from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from lmconf.config import LLMConfBase, OpenAICompatibleLLMConf
from lmconf.llm_configs.azure_openai import AzureOpenAILLMConf
from lmconf.llm_configs.tongyi import TongyiLLMConf


class NamedLLMConf(TypedDict):
    name: str
    conf: Union[AzureOpenAILLMConf, TongyiLLMConf, OpenAICompatibleLLMConf] = Field(
        discriminator="provider"
    )


class LMConfig(BaseModel):
    x: Dict[str, List[str]] = Field(default_factory=dict)
    config_list: List[NamedLLMConf] = Field(default_factory=list)

    def get(
        self,
        named_functionality: Optional[str] = None,
        named_config: Optional[str] = None,
        which_model: Optional[str] = None,
    ) -> LLMConfBase:
        """
        Args:
            named_functionality (str): the named of function which uses the specific LLM from the config.
            named_config (str): if not use named of function, use this named of the provider to get the LLMConf object.

        Returns:
            the LLMConf object specified by the name.
        """
        if not named_functionality and not named_config:
            raise ValueError("named_functionality or named_provider must be specified")
        if named_functionality:
            if named_functionality not in self.x:
                raise ValueError(f"{named_functionality} not found in lm_config.x")
            determined_llm = self.x[named_functionality]
            named_config, which_model = (
                (determined_llm[0], None) if len(determined_llm) < 2 else determined_llm
            )
        filter_ = filter(lambda d: d["name"] == named_config, self.config_list)
        default_llm_conf = list(filter_)[0]["conf"]
        if not which_model:
            return default_llm_conf
        return default_llm_conf.model_copy(update={"model": which_model})


class LMConfSettings:
    lm_config: LMConfig = Field(default_factory=LMConfig)

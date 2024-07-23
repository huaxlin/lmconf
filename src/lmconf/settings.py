from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from lmconf.config import LLMConfBase, OpenAICompatibleLLMConf
from lmconf.llm_configs.azure_openai import AzureOpenAILLMConf
from lmconf.llm_configs.tongyi import TongyiLLMConf


class NamedLLMConf(TypedDict):
    name: str
    # HACK: use TypedDict to fallback to OpenAICompatibleLLMConf
    conf: Union[AzureOpenAILLMConf, TongyiLLMConf, OpenAICompatibleLLMConf] = Field(  # type: ignore
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
        Retrieves a specific LLMs configuration object based on the provided parameters.

        Args:
            named_functionality (Optional[str]): The name of the functionality, optional.
            named_config (Optional[str]): The name of the LLMs configuration, optional.
            which_model (Optional[str]): The name of the model, optional.

        Returns:
            LLMConfBase: A base configuration object representing the large language model's configuration.

        Raises:
            ValueError: If neither `named_functionality` nor `named_config` are specified.
            ValueError: If `named_functionality` does not exist in the configuration.
        """
        # Ensure that at least one of named_functionality or named_config is specified.
        if not named_functionality and not named_config:
            raise ValueError("named_functionality or named_provider must be specified")

        # If named_functionality is specified, retrieve corresponding config information.
        if named_functionality:
            # Check if named_functionality exists within the configuration.
            if named_functionality not in self.x:
                raise ValueError(f"{named_functionality} not found in lm_config.x")

            determined_llm = self.x[named_functionality]
            # Parse the config information; if only functionality name is given, default config and model are None.
            named_config, which_model = (
                (determined_llm[0], None) if len(determined_llm) < 2 else determined_llm
            )

        # Filter the config list by named_config to obtain the matching configuration object.
        filter_ = filter(lambda d: d["name"] == named_config, self.config_list)
        default_llm_conf = list(filter_)[0]["conf"]

        # If which_model is not specified, return the default configuration object.
        if not which_model:
            return default_llm_conf

        # If which_model is specified, return a copy of the default configuration object with updated model name.
        return default_llm_conf.model_copy(update={"model": which_model})


class LMConfSettings:
    lm_config: LMConfig = Field(default_factory=LMConfig)

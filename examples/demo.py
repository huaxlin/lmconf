from pydantic_settings import BaseSettings, SettingsConfigDict
from lmconf import LMConfSettings
from rich import print


class Settings(BaseSettings, LMConfSettings):
    foo: str = ""

    model_config = SettingsConfigDict(
        env_file='demo.env',
        env_prefix='LMCONFDEMO_',
        env_nested_delimiter='__',
    )


settings = Settings()

print(settings)

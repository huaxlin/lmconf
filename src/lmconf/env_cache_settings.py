import os
from logging import getLogger
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Protocol, Type, TypeVar

if TYPE_CHECKING:
    from pydantic_settings import SettingsConfigDict

    class EnvCacheSettingsProtocol(Protocol):
        model_config: ClassVar[SettingsConfigDict]

        @classmethod
        def get_current_settings(cls): ...  # noqa: E704
        @classmethod
        def get_settings_from_env(cls): ...  # noqa: E704

else:

    class EnvCacheSettingsProtocol: ...  # noqa: E701


TBase = TypeVar("TBase", bound=EnvCacheSettingsProtocol)

logger = getLogger(__name__)
_FROM_ENV_CACHE: Dict[int, Any] = {}


class EnvCacheSettingsMixin:

    @classmethod
    # def get_current_settings(cls):
    def get_current_settings(cls: Type[TBase]) -> TBase:
        return cls.get_settings_from_env()  # type: ignore

    @classmethod
    # def get_settings_from_env(cls):
    def get_settings_from_env(cls: Type[TBase]) -> TBase:
        """
        Returns a settings object populated with default values and overrides from
        environment variables, ignoring any values in profiles.

        Calls with the same environment return a cached object instead of reconstructing
        to avoid validation overhead.
        """
        if not (env_prefix := cls.model_config["env_prefix"]):
            logger.warning(
                f"{env_prefix = }. It is recommended to set the env_prefix, "
                "as executing various third-party packages might modify environment variables, "
                "which can result in inaccurate caching that relies on environment variables."
            )
        # Since os.environ is a Dict[str, str] we can safely hash it by contents, but we
        # must be careful to avoid hashing a generator instead of a tuple
        cache_key = hash(
            tuple(
                (key, value)
                for key, value in os.environ.items()
                if key.startswith(env_prefix)  # only apply on configured env prefix
            )
        )
        if cache_key not in _FROM_ENV_CACHE:
            _FROM_ENV_CACHE[cache_key] = cls()

        return _FROM_ENV_CACHE[cache_key]

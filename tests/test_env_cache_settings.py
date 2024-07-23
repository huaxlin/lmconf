from pydantic_settings import BaseSettings, SettingsConfigDict
from lmconf.env_cache_settings import EnvCacheSettingsMixin


class Settings(BaseSettings, EnvCacheSettingsMixin):
    foo: str = ''

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='LMCONF_',
        env_nested_delimiter='__',
    )


init_settings = Settings.get_current_settings()


def test_env_cache_settings():
    import os
    from tempfile import TemporaryDirectory
    from textwrap import dedent

    assert not os.path.exists(".env")
    assert init_settings.foo == ""

    # fmt:off
    dotenv_content = dedent("""\
        LMCONF_foo = "bar"
    """)
    # fmt: on
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, ".env"), "w") as f:
            f.write(dotenv_content)
        _cwd = os.getcwd()
        os.chdir(tmpdir)

        # .env not work due to environments not changed
        first_settings = Settings.get_current_settings()
        assert first_settings is init_settings

        # !hacker: change environments will create new Settings instance
        os.environ["LMCONF_DOTENV_PATH"] = os.path.join(tmpdir, ".env")
        second_settings = Settings.get_current_settings()
        assert second_settings is not init_settings
        assert second_settings.foo == "bar"

        os.chdir(_cwd)

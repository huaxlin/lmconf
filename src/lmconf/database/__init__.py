from .db_config import DatabaseConfig  # noqa: F401


# TODO
# class DatabaseSettingsMixinProtocol: ...
# `dependencies.py`
# from contextlib import contextmanager
# def inject_db(settings_cls: Type[DatabaseSettingsMixinProtocol]):
#     def inner(fn: Callable) -> Callable:
#         @contextmanager
#         def inject(kwargs):
#             if "db" not in kwargs or kwargs["db"] is None:
#                 with settings_cls.get_current_settings().get_db() as db:
#                     kwargs["db"] = db
#                     yield
#             else:
#                 yield
#         @wraps(fn)
#         def sync_wrapper(*args, **kwargs):
#             with inject(kwargs):
#                 return fn(*args, **kwargs)
#         ...
#     return inner

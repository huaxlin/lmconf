from contextlib import contextmanager
from functools import cached_property
from typing import Generator

from pydantic import BaseModel, computed_field
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfig(BaseModel):
    SQLALCHEMY_CONNECTION_URL: str = ""

    @computed_field(repr=False)  # type: ignore
    @cached_property
    def engine(self) -> Engine:
        return create_engine(self.SQLALCHEMY_CONNECTION_URL, echo=False)

    @computed_field(repr=False)  # type: ignore
    @cached_property
    def session_maker(self) -> sessionmaker[Session]:
        return sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    @contextmanager
    def get_db(self) -> Generator[Session, None, None]:
        session_local = self.session_maker()
        try:
            yield session_local
        finally:
            session_local.close()

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

try:
    database_url = os.environ["DATABASE_URL"]
except KeyError:
    # print("Envvar DATABASE_URL not set, setting default database.")
    database_url = 'sqlite:///test.db'

if database_url.startswith("mysql"):
    engine = create_engine(database_url, convert_unicode=True,
                           pool_pre_ping=True)
else:
    engine = create_engine(database_url, convert_unicode=True)

db_session = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    import nashi.models
    Base.metadata.create_all(bind=engine)

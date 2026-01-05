from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker



SQLALCHEMY_DATABASE_URL = "postgresql://hiremeai_db_user:bgzhiVo6TC3EdKBsrOBoGgcl324cWYK7@dpg-d58pff15pdvs73dfsf6g-a.virginia-postgres.render.com/hiremeai_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(
	autocommit=False, autoflush=False, bind=engine
)
Base = declarative_base()

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

# For Neon Postgres, we'll use the database URL from settings
if settings.neon_database_url and settings.neon_database_url != "your_neon_postgres_connection_string" and settings.neon_database_url.strip() != "":
    SQLALCHEMY_DATABASE_URL = settings.neon_database_url
else:
    # Fallback to a local SQLite for development if no Neon URL provided
    SQLALCHEMY_DATABASE_URL = "sqlite:///./docusaurus_rag_chatbot.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # For Postgres/Neon, you might want to add connect_args if needed
    pool_pre_ping=True  # Validates connections before use
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    Dependency function to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Test the connection
def test_connection():
    try:
        # Try to create a session to test the connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False


# Test connection when module is imported
test_connection()
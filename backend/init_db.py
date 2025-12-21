import sys
from pathlib import Path

# Add the backend src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.database import engine, Base
from src.config.qdrant_config import initialize_qdrant_collection

def init_database():
    print("Initializing database tables...")
    try:
        # Create all tables defined in the models
        Base.metadata.create_all(bind=engine)
        print("[OK] Database tables created successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error creating database tables: {e}")
        return False

def init_qdrant():
    print("Initializing Qdrant collection...")
    try:
        # This will create the collection if it doesn't exist
        initialize_qdrant_collection()
        print("[OK] Qdrant collection initialized successfully")
        return True
    except Exception as e:
        print(f"[ERROR] Error initializing Qdrant collection: {e}")
        return False

if __name__ == "__main__":
    print("Database and Qdrant Initialization")
    print("="*40)

    db_success = init_database()
    qdrant_success = init_qdrant()

    print(f"\nInitialization Summary:")
    print(f"- Database tables: {'[OK]' if db_success else '[ERROR]'}")
    print(f"- Qdrant collection: {'[OK]' if qdrant_success else '[ERROR]'}")

    if db_success and qdrant_success:
        print("\n[OK] Both database and Qdrant are ready for document ingestion!")
    else:
        print("\n[ERROR] Initialization failed. Please check the errors above.")
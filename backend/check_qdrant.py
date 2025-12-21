import sys
from pathlib import Path

# Add the backend src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.qdrant_config import qdrant_client, settings

def check_qdrant_points():
    try:
        # Get collection info
        collection_info = qdrant_client.get_collection(collection_name=settings.qdrant_collection_name)
        print(f"Collection: {settings.qdrant_collection_name}")
        print(f"Points count: {collection_info.points_count}")

        # Get some sample points to verify they exist
        if collection_info.points_count > 0:
            # Scroll to get first few points
            points, _ = qdrant_client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=5  # Get first 5 points
            )
            print(f"\nFirst 5 points (sample):")
            for i, point in enumerate(points):
                payload = point.payload
                print(f"  {i+1}. Title: {payload.get('title', 'N/A')[:50]}... | Path: {payload.get('path', 'N/A')}")

        return collection_info.points_count
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        return 0

if __name__ == "__main__":
    print("Checking Qdrant collection...")
    count = check_qdrant_points()
    print(f"\nTotal points in collection: {count}")
import os
import sys

# src folder ko Python path mein add karo
backend_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(backend_dir, "src")
sys.path.append(src_dir)

from services.ingestion_service import IngestionService

# Project root aur docs folder
project_root = os.path.dirname(backend_dir)
docs_path = os.path.join(project_root, "docs")

print(f"Starting full book ingestion from: {docs_path}")
print("This will process all 11 markdown files in docs/ and subfolders...\n")

if not os.path.exists(docs_path):
    print(f"[ERROR] Docs folder not found: {docs_path}")
    exit()

# IngestionService banayo aur pura directory process karo
ingestion_service = IngestionService()
result = ingestion_service.process_document_directory(docs_path)

# Result print karo
print("\n=== INGESTION COMPLETE ===")
print(f"Status: {result['status']}")
print(f"Successfully processed files: {result['processed_files']}")
if result['errors']:
    print(f"Errors: {len(result['errors'])}")
    for error in result['errors']:
        print(f" - {error}")
else:
    print("No errors!")

print("\nNow check Qdrant points:")
print("Run this command: python check_qdrant.py")
print("You should see 50–300+ points depending on your book content!")
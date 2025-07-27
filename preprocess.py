import os
import shutil
from unstructured.partition.auto import partition

SOURCE_DIR = "source_documents/"
DEST_DIR = "data/"

def preprocess_documents():
    """
    Uses the Unstructured library to parse all documents in a source directory,
    extracts their text, and saves them as clean Markdown files in a destination directory.
    This acts as a normalization pipeline.
    """
    print(f"Starting pre-processing of files in '{SOURCE_DIR}'")

    # Ensure the destination directory exists
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    else:
        print(f"Cleaning destination directory: {DEST_DIR}")
        for filename in os.listdir(DEST_DIR):
            file_path = os.path.join(DEST_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    files_processed = 0
    print(f"Reading files from: {SOURCE_DIR}")
    for filename in os.listdir(SOURCE_DIR):
        source_path = os.path.join(SOURCE_DIR, filename)

        if not os.path.isfile(source_path) or filename.startswith('.'):
            continue

        print(f"Processing: {filename}...")
        try:
            # Use unstructured's partition function to handle any file type
            elements = partition(filename=source_path, strategy="auto")

            text_content = "\n\n".join([str(el) for el in elements])

            md_filename = os.path.splitext(filename)[0] + ".md"
            dest_path = os.path.join(DEST_DIR, md_filename)

            with open(dest_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            files_processed += 1
            print(f"Successfully converted and saved as: {dest_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print(f"\nPre-processing complete. Processed {files_processed} files.")

if __name__ == "__main__":
    preprocess_documents()
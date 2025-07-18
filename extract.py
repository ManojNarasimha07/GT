import os
import requests
import zipfile
import io
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Download and extract GitHub repo ---
def download_and_extract_repo(repo_url: str, extract_to: str = "repo") -> str:
    for branch in ["main", "master"]:
        try:
            zip_url = f"{repo_url.rstrip('/')}/archive/refs/heads/{branch}.zip"
            print(f"Trying {zip_url} ...")
            r = requests.get(zip_url)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
                zip_ref.extractall(extract_to)
            extracted_dirs = [d for d in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, d))]
            if not extracted_dirs:
                raise Exception("No folder found after extraction.")
            extracted_path = os.path.join(extract_to, extracted_dirs[0])
            print(f"Extracted to: {extracted_path}")
            return extracted_path
        except Exception as e:
            print(f"Failed for branch '{branch}': {e}")
    raise Exception("Failed to download repo from both 'main' and 'master' branches.")

# --- Load documents, now with .ipynb support ---
def load_documents(repo_dir: str, allowed_exts=None):
    if allowed_exts is None:
        allowed_exts = [".py", ".md", ".txt", ".json", ".yaml", ".yml", ".js", ".java", ".ts", ".css", ".html", ".csv", ".ipynb"]
    documents = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in allowed_exts):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        if file.endswith(".ipynb"):
                            # Parse notebook JSON and extract code/text cells
                            notebook = json.load(f)
                            cells = notebook.get("cells", [])
                            cell_texts = []
                            for cell in cells:
                                if cell.get("cell_type") in ["code", "markdown"]:
                                    cell_texts.append("".join(cell.get("source", [])))
                            content = "\n\n".join(cell_texts)
                        else:
                            content = f.read()
                    metadata = {"source": path}
                    documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return documents

# --- Chunk documents ---
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# --- Run ---
def run():
    x = input("ENTER THE CORRECT REPO LINK IN GITHUB: ")
    repo_url = x
    extracted_path = download_and_extract_repo(repo_url, extract_to="downloaded_repos")
    documents = load_documents(extracted_path)
    chunks = split_documents(documents)
    return documents

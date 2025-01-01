EMBEDDING_FUNCTION_NAME = "all-MiniLM-L6-v2"
ID_KEY = "doc_id"
PERSISTANT_DIRECTORY = "../data/"
PDF_DIRECTORY = "../resources/"
ALLOWED_FILE_TYPES = {"PDF": ["pdf"], "TEXT": ["txt"], "DOCX": ["docx"]}

HF_TOKEN = "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"

import os

os.environ["HF_TOKEN"] = HF_TOKEN

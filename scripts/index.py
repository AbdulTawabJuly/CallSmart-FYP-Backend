from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone, ServerlessSpec
from spacy.lang.en import English
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm.auto import tqdm
from docx2pdf import convert
from docx import Document
from typing import List
import pandas as pd
import pypandoc
import uuid
import torch
import fitz
import json
import re
import io
import os


origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MAX_FILE_SIZE = 1024 * 1024 * 50  # 50MB


# def Text_Formatter(text: str) -> str:
#     cleaned_text = text.replace("\n", " ").strip()
#     return cleaned_text


# def Open_And_Read_Pdf(content, file_type):
#     doc = fitz.open(stream=io.BytesIO(content), filetype=file_type)
#     pages_and_text = []
#     for page_number, page in enumerate(doc):
#         text = page.get_text()
#         text = Text_Formatter(text=text)
#         pages_and_text.append(
#             {
#                 "page_number": page_number,
#                 "page_char_count": len(text),
#                 "page_word_count": len(text.split(" ")),
#                 "page_sentence_count_raw": len(text.split(". ")),
#                 "page_token_count": len(text) / 4,
#                 "text": text,
#             }
#         )
#     return pages_and_text


# def Pre_Processing(pages_and_texts):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     text = ""
#     for page_number, page in enumerate(pages_and_texts):
#         text += page["text"]

#     chunks = text_splitter.split_text(text)

#     chunks_with_metadata = []
#     for i, chunk in enumerate(chunks):
#         chunks_with_metadata.append(
#             {
#                 "metadata": {
#                     "chunk_number": i,
#                     "chunk_char_count": len(chunk),
#                     "chunk_word_count": len(chunk.split(" ")),
#                     "chunk_token_count": len(chunk) / 4,
#                 },
#                 "chunk_content": chunk,
#             }
#         )
#     return chunks_with_metadata


# def Creating_Embeddings_and_Storing(chunks_with_metadata, namespace):
#     from pinecone import Pinecone

#     load_dotenv()
#     embedding_model = SentenceTransformer(
#         model_name_or_path="all-mpnet-base-v2", device="cpu"
#     )
#     print(embedding_model)
#     print("Making Embeddings")
#     for item in tqdm(chunks_with_metadata):
#         item["embeddings"] = embedding_model.encode(item["chunk_content"])
    
#     if not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_INDEX_NAME"):
#         raise HTTPException(500, "Pinecone API key or index name is missing")

#     pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     print(pinecone.list_indexes())
#     index = pinecone.Index(name=os.getenv("PINECONE_INDEX_NAME"))

#     batch_size = 100
#     batch = []

#     print("Storing the Embeddings in the Vector DB ")
#     for idx, item in enumerate(chunks_with_metadata):
#         vector = {
#             "id": str(uuid.uuid4()),
#             "values": item["embeddings"],
#             "metadata": {
#                 "text": item["chunk_content"],
#                 "metadata": json.dumps(item["metadata"]),
#             },
#         }
#         batch.append(vector)

#         if len(batch) == batch_size or idx == len(chunks_with_metadata) - 1:
#             print("Uploading")
#             try:
#                 index.upsert(vectors=batch, namespace=namespace)
#                 print("Vectors uploaded successfully")
#                 batch = []
#             except Exception as e:
#                 print(f"Error uploading vectors: {e}")

#     return "Uploaded as Vectors"


# def chunked_list(lst, n):
#     """Yield successive n-sized chunks from lst"""
#     for i in range(0, len(lst), n):
#         yield lst[i : i + n]


# def clean_data(df):
#     # Handle missing values
#     df = df.fillna("N/A")

#     # Convert datetime objects
#     for col in df.select_dtypes(include=["datetime"]).columns:
#         df[col] = df[col].dt.strftime("%Y-%m-%d")

#     # Remove special characters
#     df = df.map(lambda x: re.sub(r"[\x00-\x1F\x7F-\x9F]", "", str(x)))

#     return df


# def excel_to_search_text(df, filename):
#     text_chunks = []
#     for sheet_name, sheet_df in df.items() if isinstance(df, dict) else [("Data", df)]:
#         # Add sheet context
#         text_chunks.append(f"Data from {filename} - Sheet: {sheet_name}")

#         # Process headers
#         headers = " | ".join(sheet_df.columns)
#         text_chunks.append(f"Columns: {headers}")

#         # Convert rows to sentences
#         for idx, row in sheet_df.iterrows():
#             row_text = "Row {}: ".format(idx + 1) + ", ".join(
#                 [f"{col} is {val}" for col, val in row.items()]
#             )
#             text_chunks.append(row_text)

#     return "\n".join(text_chunks)


# def get_columns_from_chunk(chunk: str) -> list:
#     """Extract columns from chunk text"""
#     col_lines = [line for line in chunk.split("\n") if line.startswith("Columns: ")]
#     if not col_lines:
#         return []
#     return col_lines[0].replace("Columns: ", "").split(" | ")


@app.post("/scripts/data_ingestion")
async def upload_files(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    print("Hello World")
    # for file in files:
    #     content_type = file.content_type
    #     file_data = await file.read()

    #     try:
    #         # Handle PDF files
    #         if content_type == "application/pdf":
    #             print(
    #                 f"Received file: {file.filename}, Size: {round(len(file_data)/1048576,2)} MB (bytes:{len(file_data)})"
    #             )
    #             pages_and_text = Open_And_Read_Pdf(file_data, file_type="pdf")
    #             chunks_with_metadata = Pre_Processing(pages_and_text)
    #             msg = Creating_Embeddings_and_Storing(chunks_with_metadata, user_id)

    #             return {"message": f"Extracted text from {file.filename} and {msg}"}

    #         # Handle Excel files
    #         elif content_type in [
    #             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #             "application/vnd.ms-excel",
    #             "text/csv",
    #         ]:
    #             print(
    #                 f"Received file: {file.filename}, Size: {round(len(file_data)/1048576,2)} MB (bytes:{len(file_data)})"
    #             )

    #             return {
    #                     "message": f"Processed chunks from {file.filename}"
    #             }

                
    #             #     # Read with size limit
    #             #     if len(file_data) > MAX_FILE_SIZE:
    #             #         raise HTTPException(413, "File exceeds size limit")

    #             #     # Read file
    #             #     if content_type == "text/csv":
    #             #         df = pd.read_csv(io.BytesIO(file_data))
    #             #     else:
    #             #         df = pd.read_excel(io.BytesIO(file_data), sheet_name=None)

    #             #     # Clean data
    #             #     df = clean_data(df)

    #             #     # Convert to text
    #             #     full_text = excel_to_search_text(df, file.filename)

    #             #     # Split chunks
    #             #     text_splitter = RecursiveCharacterTextSplitter(
    #             #         chunk_size=1000,
    #             #         chunk_overlap=100,
    #             #         separators=["\n\n", "\n", "|", ". ", " ", ""],
    #             #     )
    #             #     chunks = text_splitter.split_text(full_text)

    #             #     embedding_model = SentenceTransformer(
    #             #         model_name_or_path="all-mpnet-base-v2", device="cpu"
    #             #     )

    #             #     pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    #             #     index = pinecone.Index(name=os.getenv("PINECONE_INDEX_NAME"))

    #             #     # Generate embeddings
    #             #     embeddings = embedding_model.encode(
    #             #         chunks, batch_size=32, show_progress_bar=True
    #             #     )

    #             #     # Prepare vectors
    #             #     vectors = [
    #             #         {
    #             #             "id": f"excel_{uuid.uuid4()}",
    #             #             "values": emb.tolist(),
    #             #             "metadata": {
    #             #                 "text": chunk,
    #             #                 "source": file.filename,
    #             #                 "data_type": "structured",
    #             #                 "columns": get_columns_from_chunk(chunk),
    #             #                 "row_count": chunk.count("Row "),
    #             #             },
    #             #         }
    #             #         for chunk, emb in zip(chunks, embeddings)
    #             #     ]

    #             #     # Batch upsert
    #             #     for batch in chunked_list(vectors, 100):
    #             #         index.upsert(vectors=batch, namespace=user_id)

    #             #     return {
    #             #         "message": f"Processed {len(chunks)} chunks from {file.filename}"
    #             #     }

    #         # Handle DOCX files
    #         elif (
    #             content_type
    #             == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    #         ):
    #             print(
    #                 f"Received file: {file.filename}, Size: {round(len(file_data)/1048576,2)} MB (bytes:{len(file_data)})"
    #             )
    #             with open("temp.docx", "wb") as f:
    #                 f.write(file_data)
    #             convert("temp.docx", "output.pdf")
    #             with open("output.pdf", "rb") as pdf_file:
    #                 pdf_content = pdf_file.read()
    #             os.remove("output.pdf")
    #             os.remove("temp.docx")

    #             pages_and_text = Open_And_Read_Pdf(pdf_content, file_type="pdf")
    #             chunks_with_metadata = Pre_Processing(pages_and_text)
    #             msg = Creating_Embeddings_and_Storing(chunks_with_metadata, user_id)

    #             return {"message": f"Extracted text from {file.filename}"}

    #         # If the file type is not supported
    #         else:
    #             raise HTTPException(
    #                 status_code=400, detail=f"Unsupported file type: {content_type}"
    #             )

    #     except Exception as e:
    #         raise HTTPException(
    #             status_code=500, detail=f"Error processing file: {str(e)}"
    #         )

import os
import pandas as pd
from typing import Any, Optional

# Import Langchain document loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredXMLLoader,
    UnstructuredEmailLoader,
    UnstructuredFileLoader,
    UnstructuredEPubLoader,
    CSVLoader,
    TextLoader
)

def get_processor_for_file(file_path: str) -> Optional[callable]:
    """
    Determine the appropriate processor function for the given file type
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Map file extensions to specific processor functions
    processors = {
        ".pdf": process_pdf,
        ".docx": process_docx,
        ".doc": process_docx,
        ".pptx": process_pptx,
        ".ppt": process_pptx,
        ".xlsx": process_xlsx,
        ".xls": process_xlsx,
        ".md": process_markdown,
        ".html": process_html,
        ".htm": process_html,
        ".xml": process_xml,
        ".msg": process_email,
        ".eml": process_email,
        ".epub": process_epub,
        ".txt": process_text,
        ".csv": process_csv,
        ".rtf": process_text,
        
        # Code files
        ".py": process_text,
        ".js": process_text,
        ".java": process_text,
        ".ts": process_text,
        ".tsx": process_text,
        ".jsx": process_text,
        ".c": process_text,
        ".cpp": process_text,
        ".h": process_text,
        ".cs": process_text,
        ".rb": process_text,
        ".go": process_text,
        ".rs": process_text,
        ".php": process_text,
        ".sql": process_text,
        ".css": process_text,
    }
    
    return processors.get(file_extension, process_generic)

def process_document(file_path: str) -> Optional[str]:
    """
    Process a document using the appropriate processor based on file type
    """
    processor = get_processor_for_file(file_path)
    if processor:
        return processor(file_path)
    return None

def process_pdf(file_path: str) -> str:
    """
    Process PDF documents using Langchain's PyMuPDFLoader
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_docx(file_path: str) -> str:
    """
    Process DOCX documents using Langchain's UnstructuredWordDocumentLoader
    """
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_pptx(file_path: str) -> str:
    """
    Process PPTX documents using Langchain's UnstructuredPowerPointLoader
    """
    loader = UnstructuredPowerPointLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_xlsx(file_path: str) -> str:
    """
    Process XLSX documents using Langchain's UnstructuredExcelLoader
    """
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_markdown(file_path: str) -> str:
    """
    Process Markdown documents using Langchain's UnstructuredMarkdownLoader
    """
    loader = UnstructuredMarkdownLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_html(file_path: str) -> str:
    """
    Process HTML documents using Langchain's UnstructuredHTMLLoader
    """
    loader = UnstructuredHTMLLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_xml(file_path: str) -> str:
    """
    Process XML documents using Langchain's UnstructuredXMLLoader
    """
    loader = UnstructuredXMLLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_email(file_path: str) -> str:
    """
    Process email documents using Langchain's UnstructuredEmailLoader
    """
    loader = UnstructuredEmailLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_text(file_path: str) -> str:
    """
    Process text documents using Langchain's TextLoader
    """
    loader = TextLoader(file_path, encoding="utf-8")
    try:
        docs = loader.load()
        
        texts = [doc.page_content for doc in docs if doc.page_content]
        combined_text = "\n\n".join(texts)
        
        return combined_text
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        loader = TextLoader(file_path, encoding="latin-1")
        docs = loader.load()
        
        texts = [doc.page_content for doc in docs if doc.page_content]
        combined_text = "\n\n".join(texts)
        
        return combined_text

def process_csv(file_path: str) -> str:
    """
    Process CSV documents using Langchain's CSVLoader
    """
    loader = CSVLoader(file_path)
    docs = loader.load()
    
    # Create a formatted string representation of the CSV data
    rows = []
    if docs:
        # Get column names from metadata if available
        if hasattr(docs[0], 'metadata') and 'columns' in docs[0].metadata:
            rows.append(",".join(docs[0].metadata['columns']))
        
        # Add content rows
        for doc in docs:
            rows.append(doc.page_content)
    
    return "\n".join(rows)

def process_epub(file_path: str) -> str:
    """
    Process EPUB documents using Langchain's UnstructuredEPubLoader
    """
    loader = UnstructuredEPubLoader(file_path)
    docs = loader.load()
    
    texts = [doc.page_content for doc in docs if doc.page_content]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_generic(file_path: str) -> str:
    """
    Generic document processor using Langchain's UnstructuredFileLoader
    """
    try:
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        
        texts = [doc.page_content for doc in docs if doc.page_content]
        combined_text = "\n\n".join(texts)
        
        return combined_text
    except Exception as e:
        # Fall back to basic text processing if UnstructuredFileLoader fails
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            # Try with a different encoding if utf-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e2:
                raise Exception(f"Could not process file: {str(e)} / {str(e2)}")

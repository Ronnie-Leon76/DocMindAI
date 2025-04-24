import os
import pymupdf4llm
import pandas as pd
import tempfile
from typing import Dict, Any, Optional, List

# Import unstructured components for different file types
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.md import partition_md
from unstructured.partition.html import partition_html
from unstructured.partition.xml import partition_xml
from unstructured.partition.email import partition_email
from unstructured.partition.text import partition_text
from unstructured.partition.epub import partition_epub

def get_processor_for_file(file_path: str) -> Optional[callable]:
    """
    Determine the appropriate processor function for the given file type
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Map file extensions to specific partition functions
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
        ".csv": process_text,
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
    Process PDF documents using pymupdf4llm
    """
    # Initialize the PDF processor
    pdf_processor = pymupdf4llm.PdfProcessor(file_path)
    
    # Extract text, tables, and images
    extracted_text = pdf_processor.extract_text()
    extracted_tables = pdf_processor.extract_tables()
    extracted_images = pdf_processor.extract_images()

    # Combine extracted content
    combined_content = []

    if extracted_text:
        combined_content.append(extracted_text)
    
    if extracted_tables:
        for table in extracted_tables:
            combined_content.append(str(table))
    
    if extracted_images:
        combined_content.append(f"Extracted {len(extracted_images)} images.")

    return "\n\n".join(combined_content)

def process_docx(file_path: str) -> str:
    """
    Process DOCX documents using unstructured
    """
    elements = partition_docx(
        filename=file_path,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_pptx(file_path: str) -> str:
    """
    Process PPTX documents using unstructured
    """
    elements = partition_pptx(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_xlsx(file_path: str) -> str:
    """
    Process XLSX documents using unstructured
    """
    elements = partition_xlsx(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_markdown(file_path: str) -> str:
    """
    Process Markdown documents using unstructured
    """
    elements = partition_md(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_html(file_path: str) -> str:
    """
    Process HTML documents using unstructured
    """
    elements = partition_html(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_xml(file_path: str) -> str:
    """
    Process XML documents using unstructured
    """
    elements = partition_xml(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_email(file_path: str) -> str:
    """
    Process email documents using unstructured
    """
    elements = partition_email(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_text(file_path: str) -> str:
    """
    Process text documents using unstructured
    """
    elements = partition_text(
        filename=file_path,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_epub(file_path: str) -> str:
    """
    Process EPUB documents using unstructured
    """
    elements = partition_epub(
        filename=file_path,
    )
    
    texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
    combined_text = "\n\n".join(texts)
    
    return combined_text

def process_generic(file_path: str) -> str:
    """
    Generic document processor using unstructured's auto partitioning
    """
    try:
        elements = partition(
            filename=file_path,
        )
        
        texts = [element.text for element in elements if hasattr(element, 'text') and element.text]
        combined_text = "\n\n".join(texts)
        
        return combined_text
    except Exception as e:
        # Fall back to basic text processing if auto-partition fails
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
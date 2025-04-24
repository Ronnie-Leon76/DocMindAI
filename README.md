# DocMind AI Installation Guide

This guide will help you set up and run DocMind AI, an open-source LLM-powered document analysis application.

## Prerequisites

1. [Python 3.8+](https://www.python.org/downloads/)
2. [Ollama](https://ollama.com/) - For running local LLMs
3. (Optional) [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/install/) for containerized deployment

## Option 1: Local Installation

1. **Clone the repository:**

```bash
git clone https://huggingface.co/spaces/davisandshirtliff/DocMindAI
cd DocMindAI
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run Ollama:**

Make sure Ollama is installed and running locally. Pull a model to use with the application:

```bash
ollama pull gemma3:1b
```

5. **Run the application:**

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501` in your web browser.

## Option 2: Docker Deployment

1. **Clone the repository:**

```bash
git clone https://huggingface.co/spaces/davisandshirtliff/DocMindAI
cd DocMindAI
```

2. **Run with Docker Compose:**

Make sure Ollama is running on your host machine, then:

```bash
docker-compose up --build
```

The application will be accessible at `http://localhost:8501` in your web browser.

## Usage

1. Enter your Ollama Base URL (default: `http://localhost:11434`)
2. Select an Ollama model from the dropdown
3. Upload documents for analysis
4. Choose your analysis settings:
   - Select a prompt type
   - Choose a tone
   - Select instructions
   - Set the desired length/detail
   - Choose the analysis mode
5. Click "Extract and Analyze"
6. Once analysis is complete, you can chat with your documents in the chat interface

## Supported File Types

DocMind AI supports a wide range of file formats including:
- PDF
- DOCX, DOC
- TXT
- XLSX, XLS
- MD (Markdown)
- JSON
- XML
- RTF
- CSV
- MSG, EML (Email)
- PPTX, PPT (PowerPoint)
- ODT (OpenDocument Text)
- EPUB (E-book)
- Code files (PY, JS, JAVA, TS, TSX, C, CPP, H, and many more)

## Troubleshooting

- If you encounter issues connecting to Ollama, make sure it's running and the URL is correct.
- For Docker deployment, ensure that your Docker configuration allows access to the host network.
- For document processing issues, check that you have the necessary dependencies installed.
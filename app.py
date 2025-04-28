import asyncio
import sys
import torch

torch.classes.__path__ = []

import streamlit as st
import pandas as pd
import os
import tempfile
from typing import List, Optional, Dict, Any, Union
import json
from datetime import datetime
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field
from Ingestion.ingest import process_document, get_processor_for_file
# Add LangChain LlamaCpp integration
from langchain_community.llms import LlamaCpp

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Set event loop policy for Windows if needed
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Set page configuration 
st.set_page_config(
    page_title="DocMind AI: AI-Powered Document Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better dark/light mode compatibility
st.markdown("""
<style>
    /* Common styles for both modes */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling for results */
    .card {
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Dark mode specific */
    @media (prefers-color-scheme: dark) {
        .card {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .highlight-container {
            background-color: rgba(255, 255, 255, 0.05);
            border-left: 3px solid #4CAF50;
        }
        
        .chat-user {
            background-color: rgba(0, 0, 0, 0.2);
        }
        
        .chat-ai {
            background-color: rgba(76, 175, 80, 0.1);
        }
    }
    
    /* Light mode specific */
    @media (prefers-color-scheme: light) {
        .card {
            background-color: rgba(0, 0, 0, 0.02);
        }
        
        .highlight-container {
            background-color: rgba(0, 0, 0, 0.03);
            border-left: 3px solid #4CAF50;
        }
        
        .chat-user {
            background-color: rgba(240, 240, 240, 0.7);
        }
        
        .chat-ai {
            background-color: rgba(76, 175, 80, 0.05);
        }
    }
    
    /* Chat message styling */
    .chat-container {
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    
    /* Highlight sections */
    .highlight-container {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Status indicators */
    .status-success {
        color: #4CAF50;
    }
    
    .status-error {
        color: #F44336;
    }
    
    /* Document list */
    .doc-list {
        list-style-type: none;
        padding-left: 0;
    }
    
    .doc-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Define the output structures using Pydantic
class DocumentAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the document")
    key_insights: List[str] = Field(description="A list of key insights from the document")
    action_items: Optional[List[str]] = Field(None, description="A list of action items derived from the document")
    open_questions: Optional[List[str]] = Field(None, description="A list of open questions or areas needing clarification")

# Function to clean up LLM responses for better parsing
def clean_llm_response(response):
    """Clean up the LLM response to extract JSON content from potential markdown code blocks."""
    # Extract content from the response
    if isinstance(response, dict) and 'choices' in response:
        content = response['choices'][0]['message']['content']
    else:
        content = str(response)
    
    # Remove markdown code block formatting if present
    if '```' in content:
        # Handle ```json format
        parts = content.split('```')
        if len(parts) >= 3:  # Has opening and closing backticks
            # Take the content between first pair of backticks
            content = parts[1]
            # Remove json language specifier if present
            if content.startswith('json') or content.startswith('JSON'):
                content = content[4:].lstrip()
    elif '`json' in content:
        # Handle `json format
        parts = content.split('`json')
        if len(parts) >= 2:
            content = parts[1]
            if '`' in content:
                content = content.split('`')[0]
    
    # Strip any leading/trailing whitespace
    content = content.strip()
    
    # Try to parse as JSON
    try:
        json_data = json.loads(content)
        
        # Check if result is nested under "properties" key
        if isinstance(json_data, dict) and "properties" in json_data:
            # Extract the properties content
            return json.dumps(json_data["properties"])
        
        return content
    except:
        # If JSON parsing fails, return the original content
        return content

# Initialize LLM without widgets in the cached function
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import os
        
        # Define model path
        model_path = "gemma-3-1b-it-q4_0_s.gguf"
        
        # Check if model exists, if not, download it
        if not os.path.exists(model_path):
            # You can use huggingface_hub to download the model
            from huggingface_hub import hf_hub_download
            
            model_path = hf_hub_download(
                repo_id="stduhpf/google-gemma-3-1b-it-qat-q4_0-gguf-small", 
                filename="gemma-3-1b-it-q4_0_s.gguf",
                cache_dir="."
            )
        
        # Use LangChain's LlamaCpp integration
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2048,
            n_ctx=4096,  # Increase context window
            top_p=1,
            verbose=False,
        )
        return llm
    except Exception as e:
        return None

# Initialize embeddings without widgets in the cached function
@st.cache_resource(show_spinner=False)
def load_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': False}
    )
    return embeddings

# Initialize session state for model loading status
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'embeddings_loaded' not in st.session_state:
    st.session_state['embeddings_loaded'] = False

# Sidebar Configuration with improved styling
st.sidebar.markdown("<div style='text-align: center;'><h1>🧠 DocMind AI</h1></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align: center;'>AI-Powered Document Analysis</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Load LLM - Only show loading spinner once
with st.sidebar:
    if not st.session_state['model_loaded']:
        with st.spinner("Loading model..."):
            llm = load_model()
            st.session_state['model_loaded'] = True
    else:
        llm = load_model()  # Will use cached version
    
    if llm is not None:
        st.markdown("<div class='status-success'>✅ Model loaded successfully!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-error'>❌ Error loading model. Check logs for details.</div>", unsafe_allow_html=True)
        st.stop()

# Load embeddings - Only show loading spinner once
with st.sidebar:
    if not st.session_state['embeddings_loaded']:
        with st.spinner("Loading embeddings..."):
            embeddings = load_embeddings()
            st.session_state['embeddings_loaded'] = True
    else:
        embeddings = load_embeddings()  # Will use cached version

# Mode Selection
with st.sidebar:
    st.markdown("### Analysis Configuration")
    analysis_mode = st.radio(
        "Analysis Mode",
        ["Analyze each document separately", "Combine analysis for all documents"]
    )

# Prompt Selection
prompt_options = {
    "Comprehensive Document Analysis": "Analyze the provided document comprehensively. Generate a summary, extract key insights, identify action items, and list open questions.",
    "Extract Key Insights and Action Items": "Extract key insights and action items from the provided document.",
    "Summarize and Identify Open Questions": "Summarize the provided document and identify any open questions that need clarification.",
    "Custom Prompt": "Enter a custom prompt below:"
}

with st.sidebar:
    st.markdown("### Prompt Settings")
    selected_prompt_option = st.selectbox("Select Prompt", list(prompt_options.keys()))
    custom_prompt = ""
    if selected_prompt_option == "Custom Prompt":
        custom_prompt = st.text_area("Enter Custom Prompt", height=100)

# Tone Selection
tone_options = [
    "Professional", "Academic", "Informal", "Creative", "Neutral", 
    "Direct", "Empathetic", "Humorous", "Authoritative", "Inquisitive"
]

with st.sidebar:
    selected_tone = st.selectbox("Select Tone", tone_options)

# Instructions Selection
instruction_options = {
    "General Assistant": "Act as a helpful assistant.",
    "Researcher": "Act as a researcher providing in-depth analysis.",
    "Software Engineer": "Act as a software engineer focusing on code and technical details.",
    "Product Manager": "Act as a product manager considering strategy and user experience.",
    "Data Scientist": "Act as a data scientist emphasizing data analysis.",
    "Business Analyst": "Act as a business analyst considering strategic aspects.",
    "Technical Writer": "Act as a technical writer creating clear documentation.",
    "Marketing Specialist": "Act as a marketing specialist focusing on branding.",
    "HR Manager": "Act as an HR manager considering people aspects.",
    "Legal Advisor": "Act as a legal advisor providing legal perspective.",
    "Custom Instructions": "Enter custom instructions below:"
}

with st.sidebar:
    st.markdown("### Assistant Behavior")
    selected_instruction = st.selectbox("Select Instructions", list(instruction_options.keys()))
    custom_instruction = ""
    if selected_instruction == "Custom Instructions":
        custom_instruction = st.text_area("Enter Custom Instructions", height=100)

# Length/Detail Selection
length_options = ["Concise", "Detailed", "Comprehensive", "Bullet Points"]

with st.sidebar:
    st.markdown("### Response Format")
    selected_length = st.selectbox("Select Length/Detail", length_options)

# Main Area
st.markdown("<h1 style='text-align: center;'>📄 DocMind AI: Document Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload documents and analyze them using the Gemma model</p>", unsafe_allow_html=True)

# File Upload with improved UI
uploaded_files = st.file_uploader(
    "Upload Documents", 
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "xlsx", "md", "json", "xml", "rtf", "csv", "msg", "pptx", "odt", "epub", 
          "py", "js", "java", "ts", "tsx", "c", "cpp", "h", "html", "css", "sql", "rb", "go", "rs", "php"]
)

# Display uploaded files with better visual indication
if uploaded_files:
    st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
    st.markdown("### Uploaded Documents")
    st.markdown("<ul class='doc-list'>", unsafe_allow_html=True)
    for file in uploaded_files:
        st.markdown(f"<li>📄 {file.name}</li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Function to truncate document to fit context window
def truncate_to_fit_context(text, max_tokens=3500):
    """Truncate text to approximately fit within context window."""
    # Very simple approximation: 4 chars ~= 1 token
    approx_tokens = len(text) / 4
    if approx_tokens > max_tokens:
        char_limit = max_tokens * 4
        return text[:int(char_limit)] + "\n\n[Document truncated due to length...]"
    return text

# Function to process the documents and run analysis
def run_analysis():
    if not uploaded_files:
        st.error("Please upload at least one document.")
        return
    
    # Save uploaded files to temporary directory
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    # Process documents
    with st.spinner("Processing documents..."):
        all_texts = []
        processed_docs = []
        
        progress_bar = st.progress(0)
        for i, file_path in enumerate(file_paths):
            processor = get_processor_for_file(file_path)
            if processor:
                try:
                    doc_data = process_document(file_path)
                    if doc_data is not None and len(doc_data.strip()) > 0:  # Ensure we have content
                        # Truncate document if too large
                        doc_data = truncate_to_fit_context(doc_data, max_tokens=3000)  # Reduced for safety
                        all_texts.append(doc_data)
                        processed_docs.append({"name": os.path.basename(file_path), "data": doc_data})
                except Exception as e:
                    st.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            progress_bar.progress((i + 1) / len(file_paths))
    
    if not all_texts:
        st.error("No documents could be processed. Please check the file formats and try again.")
        return
    
    # Build the prompt
    if selected_prompt_option == "Custom Prompt":
        prompt_text = custom_prompt
    else:
        prompt_text = prompt_options[selected_prompt_option]
    
    if selected_instruction == "Custom Instructions":
        instruction_text = custom_instruction
    else:
        instruction_text = instruction_options[selected_instruction]
    
    # Add tone guidance
    tone_guidance = f"Use a {selected_tone.lower()} tone in your response."
    
    # Add length guidance
    length_guidance = ""
    if selected_length == "Concise":
        length_guidance = "Keep your response brief and to the point."
    elif selected_length == "Detailed":
        length_guidance = "Provide a detailed response with thorough explanations."
    elif selected_length == "Comprehensive":
        length_guidance = "Provide a comprehensive in-depth analysis covering all aspects."
    elif selected_length == "Bullet Points":
        length_guidance = "Format your response primarily using bullet points for clarity."
    
    # Set up the output parser
    output_parser = PydanticOutputParser(pydantic_object=DocumentAnalysis)
    format_instructions = output_parser.get_format_instructions()
    
    if analysis_mode == "Analyze each document separately":
        results = []
        
        for doc in processed_docs:
            with st.spinner(f"Analyzing {doc['name']}..."):
                # Create system message with combined instructions
                system_message = f"{instruction_text} {tone_guidance} {length_guidance} Format your response according to these instructions: {format_instructions}"
                
                prompt = f"""
                {prompt_text}
                Document: {doc['name']}
                Content: {doc['data']}
                """
                
                # Use LlamaCpp directly through langchain
                try:
                    # Create a prompt template for the LLM
                    system_template = f"{instruction_text} {tone_guidance} {length_guidance}"
                    messages = [
                        SystemMessage(content=system_template),
                        SystemMessage(content=f"Format your response according to these instructions: {format_instructions}"),
                        HumanMessage(content="{input}")
                    ]
                    template = ChatPromptTemplate.from_messages(messages)
                    
                    # Get response from LLM
                    chain = template | llm
                    response = chain.invoke({"input": prompt})
                    
                    # Try to parse the response into the pydantic model
                    try:
                        # Clean the response before parsing
                        cleaned_response = clean_llm_response(response)
                        parsed_response = output_parser.parse(cleaned_response)
                        results.append({
                            "document_name": doc['name'],
                            "analysis": parsed_response.dict()
                        })
                    except Exception as e:
                        # If parsing fails, include the raw response
                        results.append({
                            "document_name": doc['name'],
                            "analysis": str(response),
                            "parsing_error": str(e)
                        })
                except Exception as e:
                    st.error(f"Error analyzing {doc['name']}: {str(e)}")
        
        # Display results with card-based UI
        for result in results:
            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Analysis for: {result['document_name']}</h3>", unsafe_allow_html=True)
            
            if isinstance(result['analysis'], dict) and 'parsing_error' not in result:
                # Structured output
                st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                st.markdown("### Summary")
                st.write(result['analysis']['summary'])
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("### Key Insights")
                for insight in result['analysis']['key_insights']:
                    st.markdown(f"- {insight}")
                
                if result['analysis'].get('action_items'):
                    st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                    st.markdown("### Action Items")
                    for item in result['analysis']['action_items']:
                        st.markdown(f"- {item}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if result['analysis'].get('open_questions'):
                    st.markdown("### Open Questions")
                    for question in result['analysis']['open_questions']:
                        st.markdown(f"- {question}")
            else:
                # Raw output
                st.markdown(result['analysis'])
                if 'parsing_error' in result:
                    st.info(f"Note: The response could not be parsed into the expected format. Error: {result['parsing_error']}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        with st.spinner("Analyzing all documents together..."):
            # Combine all documents - limit total size
            combined_docs = []
            total_size = 0
            max_size = 10000  # Approximate token limit for combined docs
            
            for doc in processed_docs:
                doc_content = f"Document: {doc['name']}\n\nContent: {doc['data']}"
                # Rough token estimate (4 chars per token)
                doc_tokens = len(doc_content) / 4
                
                if total_size + doc_tokens <= max_size:
                    combined_docs.append(doc_content)
                    total_size += doc_tokens
                else:
                    # Add notification that some documents were excluded
                    combined_docs.append(f"Document: {doc['name']}\n\nContent: [Document excluded due to total size limits]")
            
            combined_content = "\n\n".join(combined_docs)
            
            # Create system message with combined instructions
            system_message = f"{instruction_text} {tone_guidance} {length_guidance} Format your response according to these instructions: {format_instructions}"
            
            # Create the prompt template 
            template = ChatPromptTemplate.from_messages([
                ("system", system_message),
                ("human", "{input}")
            ])
            
            # Create the prompt
            prompt = f"""
            {prompt_text}
            {combined_content}
            """
            
            # Get response from LLM
            try:
                chain = template | llm
                response = chain.invoke({"input": prompt})
                
                # Try to parse the response into the pydantic model
                try:
                    # Clean the response before parsing
                    cleaned_response = clean_llm_response(response)
                    parsed_response = output_parser.parse(cleaned_response)
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>Combined Analysis for All Documents</h3>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                    st.markdown("### Summary")
                    st.write(parsed_response.summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("### Key Insights")
                    for insight in parsed_response.key_insights:
                        st.markdown(f"- {insight}")
                    
                    if parsed_response.action_items:
                        st.markdown("<div class='highlight-container'>", unsafe_allow_html=True)
                        st.markdown("### Action Items")
                        for item in parsed_response.action_items:
                            st.markdown(f"- {item}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    if parsed_response.open_questions:
                        st.markdown("### Open Questions")
                        for question in parsed_response.open_questions:
                            st.markdown(f"- {question}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    # If parsing fails, return the raw response
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<h3>Combined Analysis for All Documents</h3>", unsafe_allow_html=True)
                    st.markdown(str(response))
                    st.info(f"Note: The response could not be parsed into the expected format. Error: {str(e)}")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error analyzing documents: {str(e)}")
    
    # Create text chunks for embeddings
    with st.spinner("Setting up document chat..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks to prevent token overflow
                chunk_overlap=50
            )
            
            all_chunks = []
            for doc in processed_docs:
                if doc['data'] and len(doc['data'].strip()) > 0:  # Verify data exists and is not empty
                    chunks = text_splitter.split_text(doc['data'])
                    all_chunks.extend(chunks)
            
            # Only create embeddings if we have chunks
            if all_chunks and len(all_chunks) > 0:
                # Using 'None' as namespace to avoid unique ID issues with Chroma
                vectorstore = Chroma.from_texts(
                    texts=all_chunks, 
                    embedding=embeddings,
                    collection_name="docmind_collection",
                    collection_metadata={"timestamp": datetime.now().isoformat()}
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 chunks
                
                # Create a custom wrapper function for the conversational retrieval chain
                def conversational_qa(query):
                    # Get relevant documents
                    docs = retriever.get_relevant_documents(query)
                    
                    # Extract text from documents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Create a simple prompt template
                    system_template = """You are a helpful AI assistant that answers questions about documents.
                    Use the following pieces of retrieved context to answer the user's question.
                    If you don't know the answer, just say you don't know.
                    
                    Context:
                    {context}
                    """
                    
                    # Combine context and query using LLM directly
                    template = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("human", "{question}")
                    ])
                    
                    # Process with LLM
                    response = template.invoke({
                        "context": context,
                        "question": query
                    }) | llm
                    
                    return {"answer": response}
                
                # Store the function in session state
                st.session_state['qa_function'] = conversational_qa
                st.session_state['chat_history'] = []
                
                st.success("Document chat is ready! Ask questions about your documents below.")
            else:
                st.warning("No text chunks were created from the documents. Chat functionality is unavailable.")
        
        except Exception as e:
            st.error(f"Error setting up document chat: {str(e)}")
            # For debugging purposes
            st.exception(e)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chat Interface with improved styling
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>💬 Chat with your Documents</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask follow-up questions about the analyzed documents.</p>", unsafe_allow_html=True)

# Process the analysis if button is clicked
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Extract and Analyze", use_container_width=True):
        run_analysis()

# Chat input and display
if 'qa_function' in st.session_state:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state['qa_function'](user_question)
                st.session_state['chat_history'].append({"question": user_question, "answer": response['answer']})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display chat history with improved styling
    for exchange in st.session_state['chat_history']:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message chat-user'><strong>You:</strong> {exchange['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-message chat-ai'><strong>DocMind AI:</strong> {exchange['answer']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
    <p>Built with ❤️ using Streamlit, LangChain, and Gemma model</p>
    <p>DocMind AI - AI-Powered Document Analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
import os
import tempfile
from typing import List, Optional, Dict, Any, Union
import json
from datetime import datetime
from llama_cpp import Llama
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from Ingestion.ingest import process_document, get_processor_for_file

# Set page configuration
st.set_page_config(
    page_title="DocMind AI: AI-Powered Document Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the output structures using Pydantic
class DocumentAnalysis(BaseModel):
    summary: str = Field(description="A concise summary of the document")
    key_insights: List[str] = Field(description="A list of key insights from the document")
    action_items: Optional[List[str]] = Field(None, description="A list of action items derived from the document")
    open_questions: Optional[List[str]] = Field(None, description="A list of open questions or areas needing clarification")

# Initialize LLM and Model Cache
@st.cache_resource
def load_model():
    llm = Llama.from_pretrained(
        repo_id="google/gemma-3-4b-it-qat-q4_0-gguf",
        filename="gemma-3-4b-it-q4_0.gguf",
    )

    return llm

# Sidebar Configuration
st.sidebar.title("🧠 DocMind AI")
st.sidebar.markdown("AI-Powered Document Analysis")
st.sidebar.markdown("---")

# Load LLM
with st.spinner("Loading the Gemma 1.1B model..."):
    try:
        llm = load_model()
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Mode Selection
analysis_mode = st.sidebar.radio(
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
selected_prompt_option = st.sidebar.selectbox("Select Prompt", list(prompt_options.keys()))
custom_prompt = ""
if selected_prompt_option == "Custom Prompt":
    custom_prompt = st.sidebar.text_area("Enter Custom Prompt", height=100)

# Tone Selection
tone_options = [
    "Professional", "Academic", "Informal", "Creative", "Neutral", 
    "Direct", "Empathetic", "Humorous", "Authoritative", "Inquisitive"
]
selected_tone = st.sidebar.selectbox("Select Tone", tone_options)

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
selected_instruction = st.sidebar.selectbox("Select Instructions", list(instruction_options.keys()))
custom_instruction = ""
if selected_instruction == "Custom Instructions":
    custom_instruction = st.sidebar.text_area("Enter Custom Instructions", height=100)

# Length/Detail Selection
length_options = ["Concise", "Detailed", "Comprehensive", "Bullet Points"]
selected_length = st.sidebar.selectbox("Select Length/Detail", length_options)

# Main Area
st.title("📄 DocMind AI: Document Analysis")
st.markdown("Upload documents and analyze them using the Gemma 1.1B language model.")

# File Upload
uploaded_files = st.file_uploader(
    "Upload Documents", 
    accept_multiple_files=True,
    type=["pdf", "docx", "txt", "xlsx", "md", "json", "xml", "rtf", "csv", "msg", "pptx", "odt", "epub", 
          "py", "js", "java", "ts", "tsx", "c", "cpp", "h", "html", "css", "sql", "rb", "go", "rs", "php"]
)

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
        
        for file_path in file_paths:
            processor = get_processor_for_file(file_path)
            if processor:
                try:
                    doc_data = process_document(file_path)
                    if doc_data is not None:
                        all_texts.append(doc_data)
                        processed_docs.append({"name": os.path.basename(file_path), "data": doc_data})
                except Exception as e:
                    st.error(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
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
                
                # Get response from LLM
                try:
                    response = llm.create_chat_completion(
                        messages = [
                            {
                                "role": "system",
                                "content": system_message
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )
                    # Try to parse the response into the pydantic model
                    try:
                        parsed_response = output_parser.parse(response)
                        results.append({
                            "document_name": doc['name'],
                            "analysis": parsed_response.dict()
                        })
                    except Exception as e:
                        results.append({
                            "document_name": doc['name'],
                            "analysis": response,
                            "parsing_error": str(e)
                        })
                except Exception as e:
                    st.error(f"Error analyzing {doc['name']}: {str(e)}")
        
        # Display results
        for result in results:
            st.subheader(f"Analysis for: {result['document_name']}")
            
            if isinstance(result['analysis'], dict) and 'parsing_error' not in result:
                # Structured output
                st.markdown("### Summary")
                st.write(result['analysis']['summary'])
                
                st.markdown("### Key Insights")
                for insight in result['analysis']['key_insights']:
                    st.markdown(f"- {insight}")
                
                if result['analysis'].get('action_items'):
                    st.markdown("### Action Items")
                    for item in result['analysis']['action_items']:
                        st.markdown(f"- {item}")
                
                if result['analysis'].get('open_questions'):
                    st.markdown("### Open Questions")
                    for question in result['analysis']['open_questions']:
                        st.markdown(f"- {question}")
            else:
                # Raw output
                st.markdown(result['analysis'])
                if 'parsing_error' in result:
                    st.info(f"Note: The response could not be parsed into the expected format. Error: {result['parsing_error']}")
    
    else:
        with st.spinner("Analyzing all documents together..."):
            # Combine all documents
            combined_content = "\n\n".join([f"Document: {doc['name']}\n\nContent: {doc['data']}" for doc in processed_docs])
            
            # Create system message with combined instructions
            system_message = f"{instruction_text} {tone_guidance} {length_guidance} Format your response according to these instructions: {format_instructions}"
            
            # Create the prompt template for HuggingFace models
            prompt = f"""
            {prompt_text}
            {combined_content}
            """
            
            # Get response from LLM
            try:
                response = llm.create_chat_completion(
                    messages = [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
                
                # Try to parse the response into the pydantic model
                try:
                    parsed_response = output_parser.parse(response)
                    st.subheader("Combined Analysis for All Documents")
                    
                    st.markdown("### Summary")
                    st.write(parsed_response.summary)
                    
                    st.markdown("### Key Insights")
                    for insight in parsed_response.key_insights:
                        st.markdown(f"- {insight}")
                    
                    if parsed_response.action_items:
                        st.markdown("### Action Items")
                        for item in parsed_response.action_items:
                            st.markdown(f"- {item}")
                    
                    if parsed_response.open_questions:
                        st.markdown("### Open Questions")
                        for question in parsed_response.open_questions:
                            st.markdown(f"- {question}")
                
                except Exception as e:
                    # If parsing fails, return the raw response
                    st.subheader("Combined Analysis for All Documents")
                    st.markdown(response)
                    st.info(f"Note: The response could not be parsed into the expected format. Error: {str(e)}")
            
            except Exception as e:
                st.error(f"Error analyzing documents: {str(e)}")
    
    # Set up chat with documents
    with st.spinner("Setting up document chat..."):
        try:
            # Create text chunks for embeddings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            all_chunks = []
            for doc in processed_docs:
                chunks = text_splitter.split_text(doc['data'])
                all_chunks.extend(chunks)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            vectorstore = Chroma.from_texts(texts=all_chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # Set up conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create conversational chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory
            )
            
            st.session_state['qa_chain'] = qa_chain
            st.session_state['chat_history'] = []
        
        except Exception as e:
            st.error(f"Error setting up document chat: {str(e)}")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Chat Interface
st.markdown("---")
st.subheader("💬 Chat with your Documents")
st.markdown("Ask follow-up questions about the analyzed documents.")

# Process the analysis if button is clicked
if st.button("Extract and Analyze"):
    run_analysis()

# Chat input and display
if 'qa_chain' in st.session_state:
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state['qa_chain'].invoke({"question": user_question})
                st.session_state['chat_history'].append({"question": user_question, "answer": response['answer']})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Display chat history
    for exchange in st.session_state['chat_history']:
        st.markdown(f"**You:** {exchange['question']}")
        st.markdown(f"**DocMind AI:** {exchange['answer']}")
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
    <p>Built with ❤️ using Streamlit, LangChain, and Gemma 1.1B</p>
    <p>DocMind AI - AI-Powered Document Analysis</p>
    </div>
    """, 
    unsafe_allow_html=True
)
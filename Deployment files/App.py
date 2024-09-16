import os
# Ensure the static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

import io
import re
from pdfminer.high_level import extract_text
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# Initialize the models and embeddings
primary_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key='AIzaSyCHfQMVzQHsGDTBSv4NCadj1oLdPVtMx80',
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key='AIzaSyCHfQMVzQHsGDTBSv4NCadj1oLdPVtMx80'
)

fallback_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key='AIzaSyCHfQMVzQHsGDTBSv4NCadj1oLdPVtMx80',
    temperature=0.5,
    convert_system_message_to_human=True
)

# Global list to store all conversations
conversation_history = []

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file using pdfminer.six."""
    try:
        file_stream.seek(0)  # Ensure the stream is at the beginning
        text = extract_text(file_stream)
        if not text.strip():
            print("Extracted text is empty.")
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")  # Debug: Log extraction error
        return ""

def validate_pdf(file_stream):
    """Validate if the uploaded file is a valid PDF by extracting text."""
    try:
        text = extract_text_from_pdf(file_stream)
        if not text.strip():
            print("No text found in the PDF.")  # Debug: Log if no text is found
        return bool(text.strip())
    except Exception as e:
        print(f"PDF validation error: {e}")  # Debug: Log validation error
        return False

def load_pdf(file_stream):
    """Load PDF content."""
    try:
        text = extract_text_from_pdf(file_stream)
        if text.strip():
            return [text]
        else:
            return ["No text found in the PDF."]
    except Exception as e:
        print(f"Error reading the PDF file: {e}")  # Debug: Log read error
        return [f"Error reading the PDF file: {e}"]

def get_fallback_answer(question):
    """Get an answer using the fallback model if the PDF content is not sufficient."""
    try:
        # Construct a prompt for the fallback model
        prompt = (
            "Please provide a comprehensive answer to the following question based on general knowledge:\n\n"
            f"Question: {question}"
        )
        # Perform the query using the fallback model
        response = fallback_model.invoke(prompt)
        
        # Ensure the response is in a text format
        if isinstance(response, dict):
            return response.get("content", "No answer found.")
        elif hasattr(response, 'content'):
            return response.content  # Adjust based on the actual response object structure
        else:
            return str(response)  # Fallback to string conversion if response structure is unknown
    except Exception as e:
        print(f"An error occurred with the fallback model: {e}")  # Debug: Log error
        return f"An error occurred with the fallback model: {e}"


def clean_text(text):
    """Clean up unwanted symbols and formatting from the text."""
    # Remove ** and other unwanted symbols
    text = re.sub(r'\*\*', '', text)
    return text

def generate_pdf(conversation, output_path):
    """Generate a PDF from the conversation text."""
    
    # Create a PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    
    # Define styles
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    
    # Prepare the content for the PDF
    content = []
    
    for line in conversation:
        # Clean up the line
        clean_line = clean_text(line)
        # Add a paragraph to the content with line wrapping
        para = Paragraph(clean_line, normal_style)
        content.append(para)
        
        # Add a spacer between conversations
        content.append(Spacer(1, 12))  # 12 points space between entries
    
    # Build the PDF
    doc.build(content)

def process_pdf_and_answer_question(pdf_file, question):
    try:
        # Convert bytes to a file-like object
        file_stream = io.BytesIO(pdf_file)

        # Validate the PDF
        if not validate_pdf(file_stream):
            return "The uploaded file is not a valid PDF.", None

        # Reset file pointer to the beginning after validation
        file_stream.seek(0)

        pages = load_pdf(file_stream)

        if isinstance(pages, str):
            return pages, None

        context = "\n\n".join(pages)  # Combine all page contents into a single string

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        texts = text_splitter.split_text(context)

        # Create vector index
        vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

        # Initialize the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            primary_model,
            retriever=vector_index,
            return_source_documents=True
        )

        # Construct the prompt with additional context
        prompt = (
            "You are an expert in answering questions based on provided content and additional knowledge with suitable heading. Suggest something you know from your base knowledge if the user asks. "
            "Please provide a comprehensive answer based on the following PDF content and your own knowledge base.\n\n"
            f"PDF Content:\n{context}\n\n"
            f"Question: {question}"
        )

        # Perform the query
        result = qa_chain({"query": prompt})

        # Retrieve the answer from the PDF content
        pdf_answer = result.get("result", None)

        # Check if the primary model's answer indicates inability to answer
        if pdf_answer and any(phrase in pdf_answer.lower() for phrase in ["i cannot", "i don't", "no answer found"]):
            pdf_answer = get_fallback_answer(question)

        # Append the current interaction to the conversation history
        conversation_history.append(f"Question: {question}")
        conversation_history.append(f"Answer: {pdf_answer}")

        # Generate PDF for the conversation history
        output_path = "static/conversation_history.pdf"
        generate_pdf(conversation_history, output_path)

        # Return the answer and the path to the generated PDF
        return pdf_answer, output_path

    except Exception as e:
        print(f"An error occurred: {e}")  # Debug: Log error
        return f"An error occurred: {e}", None

# Define Gradio interface with download link
iface = gr.Interface(
    fn=process_pdf_and_answer_question,
    inputs=[
        gr.File(label="Upload PDF", type="binary"),  # 'binary' type for direct bytes handling
        gr.Textbox(label="Ask a Question", lines=2)
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.File(label="Download PDF")
    ],
    title="PDF Q&A System",
    description="Upload a PDF and ask questions about its content. The system will provide comprehensive answers using both the PDF content and additional knowledge."
)

# Launch the Gradio app
iface.launch()

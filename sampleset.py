

import gradio as gr
import pdfplumber
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import io

# Initialize the models and embeddings
primary_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key='AIzaSyAapf_GYO6P5XZz8Kp9b8rh-25eKV7UYt8',
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key='AIzaSyAapf_GYO6P5XZz8Kp9b8rh-25eKV7UYt8'
)

fallback_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key='AIzaSyAapf_GYO6P5XZz8Kp9b8rh-25eKV7UYt8',
    temperature=0.5,
    convert_system_message_to_human=True
)

def validate_pdf(file):
    """Validate if the uploaded file is a valid PDF."""
    try:
        with fitz.open(stream=file, filetype='pdf') as pdf:
            if pdf.page_count > 0:
                return True
    except Exception:
        pass
    return False

def load_pdf(file):
    pages = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
                else:
                    pages.append("No text found on this page.")
    except Exception as e:
        return f"Error reading the PDF file: {e}"
    return pages

def get_fallback_answer(question):
    """Get an answer using the fallback model if the PDF content is not sufficient."""
    try:
        # Construct a prompt for the fallback model
        prompt = (
            "Please provide a comprehensive answer to the following question based on general knowledge:\n\n"
            f"Question: {question}"
        )
        # Perform the query using the fallback model
        result = fallback_model.invoke(prompt)
        return result.get("content", "No answer found.")
    except Exception as e:
        return f"An error occurred with the fallback model: {e}"

def process_pdf_and_answer_question(pdf_file, question):
    try:
        # Convert bytes to a file-like object
        pdf_file = io.BytesIO(pdf_file)

        # Validate the PDF
        if not validate_pdf(pdf_file):
            return "The uploaded file is not a valid PDF."

        # Reset file pointer to the beginning after validation
        pdf_file.seek(0)

        pages = load_pdf(pdf_file)

        if isinstance(pages, str):
            return pages

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

        # If no answer is found from the PDF content, use the fallback model
        if not pdf_answer or pdf_answer == "No answer found.":
            pdf_answer = get_fallback_answer(question)

        # Return the answer
        return pdf_answer

    except Exception as e:
        return f"An error occurred: {e}"

# Define Gradio interface without examples
iface = gr.Interface(
    fn=process_pdf_and_answer_question,
    inputs=[
        gr.File(label="Upload PDF", type="binary"),  # 'binary' type for direct bytes handling
        gr.Textbox(label="Ask a Question", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Q&A System",
    description="Upload a PDF and ask questions about its content. The system will provide comprehensive answers using both the PDF content and additional knowledge."
)

# Launch the Gradio app
iface.launch()
[LIVE RAG APP][https://huggingface.co/spaces/Inela/RAGBOT]



#RAGBOT 


1.Overview


     The PDF Q&A System allows users to upload a PDF document and ask questions about its content. The system leverages advanced language models and embeddings to provide comprehensive answers based on the provided PDF and additional general knowledge. It also maintains a history of conversations, which can be downloaded as a PDF.

2.Features


     1.PDF Upload: Allows users to upload a PDF document for processing.
     2.Question Answering: Answers questions based on the content of the uploaded PDF.
     3.Fallback Model: Provides answers using a general knowledge model if the primary model cannot provide a satisfactory answer.
     4.Conversation History: Keeps a history of questions and answers, which can be downloaded as a PDF document.

3.Technologies
     
     
     1.Python: The primary programming language used for development.
     2.pdfminer.six: For extracting text from PDF files.
     3.Gradio: For creating the web interface.
     4.Langchain: For text splitting, vector storage(Chroma DB), and retrieval-based question answering.
     5.Google Generative AI: For generating answers based on content and general knowledge.
     6.ReportLab: For generating PDFs from text content.

4.Setup


     API Keys           :Replace google_api_key in the script with your actual Google API key.
     Directory Structure:The script will create the 'static' directory if it does not already exist.

5.Code Explanation


     Text Extraction and Validation:
            The extract_text_from_pdf function extracts text from the uploaded PDF.
            The validate_pdf function ensures the PDF contains extractable text.

     Question Answering:
            The process_pdf_and_answer_question function handles the logic for processing the PDF, generating answers, and managing conversation history.

     Fallback Mechanism:
            If the primary model cannot provide a satisfactory answer, the get_fallback_answer function queries a fallback model for a general knowledge-based answer.

     PDF Generation:
            The generate_pdf function creates a PDF document from the conversation history.


6.Deployment



     Running in Huggingface Space

     

# RAG-BOT-Chat-with-PDF


   Initialization
   
        Models and Embeddings:
             Primary_model ---An instance of ChatGoogleGenerativeAI with a specific model configuration (gemini-pro). This 
                              model is used for generating answers based on the content of the PDF and additional 
                              knowledge.
             Embeddings    ---An instance of GoogleGenerativeAIEmbeddings for generating text embeddings from the PDF 
                              content.
             Fallback_model---Another instance of ChatGoogleGenerativeAI with different temperature settings for fallback 
                              scenarios when the primary model cannot provide an answer.


          
  
   
   
   Functions
   
        Validate_pdf(file)
             Purpose: Checks if the uploaded file is a valid PDF.
             Method: Uses fitz (PyMuPDF) to open the file and verify that it has at least one page.
             Returns: True if valid, otherwise False.
        Load_pdf(file)
             Purpose: Extracts text from each page of the PDF.
             Method : Utilizes pdfplumber to open the PDF and extract text. If a page contains no text, it appends a 
                      placeholder message.
             Returns: A list of text strings, one per page, or an error message if the PDF cannot be read.
        Get_fallback_answer(question)
             Purpose: Provides an answer using a fallback model if the primary model fails to generate a satisfactory 
                      response.
             Method: Constructs a prompt for the fallback model and queries it.
             Returns: The answer from the fallback model or an error message if the model fails.
        Process_pdf_and_answer_question(pdf_file, question)
             Purpose: Manages the complete process of validating, loading, and querying the PDF.
             Method:
                    Converts the PDF file to a file-like object.
                    Validates and loads the PDF content.
                    Joins the content of all pages into a single string.
                    Splits the text into chunks for processing.
                    Creates a vector index using Chroma from the text chunks.
                    Initializes a RetrievalQA chain with the primary model and vector index.
                    Constructs and executes a query to generate an answer.
                    If the primary model does not provide an answer, uses the fallback model.
             Returns: The generated answer based on the PDF content and/or fallback model.



   
   
   
   
   Gradio Interface
   
         Interface Definition:
              Inputs:
                   gr.File: For uploading the PDF file.
                   gr.Textbox: For entering questions about the PDF content.
              Outputs:
                   gr.Textbox: Displays the generated answer.
                   Title and Description: Provides context about the PDF Q&A system.
         Launch:
              Method: Calls iface.launch() to start the Gradio app, making it accessible through a web browser.

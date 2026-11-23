import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Read multiple PDFs and combine their text
pdf_paths = ["document1.pdf", "document2.pdf"]  # add your PDF file paths here
documents = []
for pdf in pdf_paths:
    text = extract_text_from_pdf(pdf)
    documents.append(text)

# Split the documents into smaller chunks for better context management
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = []
for doc in documents:
    docs.extend(text_splitter.split_text(doc))

# Create embeddings and build a vector store (FAISS)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)

# Create a RetrievalQA chain using the vector store and a generative model
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=retriever)

# Now, you can ask questions:
question = "What are the key findings of the documents?"
answer = qa_chain.run(question)
print("Answer:", answer)

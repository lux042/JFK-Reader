import os
import json
import io
import fitz  # PyMuPDF
import pdf2image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

# Paths
PDF_FOLDER = os.path.expanduser("~/Documents/queryai/JFK_Documents_Test")
OUTPUT_FILE = "document_analysis.json"

# Initialize AI Models
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
layoutlm_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
layoutlm_model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Questions for analysis
questions = [
    "what is the main concept?"
]

# Function to extract text using TrOCR
def extract_text_trocr(pdf_path):
    text = ""
    
    # Convert PDF pages to images
    for page_img in pdf2image.convert_from_path(pdf_path, dpi=300):
        page_img = page_img.convert("RGB")  # Ensure image is in RGB mode
        pixel_values = ocr_processor(images=page_img, return_tensors="pt").pixel_values

        # Perform OCR
        generated_text = ocr_model.generate(pixel_values)
        decoded_text = ocr_processor.batch_decode(generated_text, skip_special_tokens=True)[0]

        text += decoded_text + "\n"

    return text.strip()

# Fallback OCR using PyMuPDF
def extract_text_pymupdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Function to perform question answering with LayoutLMv3
def get_answers_from_text(text, questions):
    answers = {}
    for question in questions:
        try:
            inputs = layoutlm_processor(question=question, context=text, return_tensors="pt")
            outputs = layoutlm_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = inputs["input_ids"][0][answer_start:answer_end]
            answers[question] = layoutlm_processor.decode(answer)
        except Exception:
            answers[question] = "No answer found"
    return answers

# Function to compare documents using Sentence-BERT
def compare_documents(texts):
    if len(texts) < 2:
        return []

    embeddings = similarity_model.encode(texts)
    num_docs = len(texts)
    similarity_matrix = [[0] * num_docs for _ in range(num_docs)]

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity

    return similarity_matrix

# Function to convert numpy float32 to Python float
def float32_to_float(o):
    if isinstance(o, np.float32):
        return float(o)  # Convert np.float32 to a native float
    raise TypeError(f'Object of type {o.__class__.__name__} is not serializable')

# Main processing function
def process_pdfs(pdf_folder):
    document_data = []

    if not os.path.exists(pdf_folder):
        print(f"Error: Folder '{pdf_folder}' does not exist.")
        return []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")

            # Extract text using TrOCR first
            extracted_text = extract_text_trocr(pdf_path)
            
            # If TrOCR fails, fall back to PyMuPDF
            if not extracted_text:
                extracted_text = extract_text_pymupdf(pdf_path)

            if not extracted_text:
                continue

            # Get answers
            answers = get_answers_from_text(extracted_text, questions)

            # Store results
            document_data.append({
                "filename": filename,
                "text": extracted_text[:1000] + "...",  # Store only first 1000 characters for preview
                "answers": answers
            })

    return document_data

# Run the processing
documents = process_pdfs(PDF_FOLDER)

# Compare document similarity if there are multiple documents
texts = [doc["text"] for doc in documents]
similarity_matrix = compare_documents(texts) if len(documents) > 1 else []

# Save results
output_data = {
    "documents": documents,
    "similarity_matrix": similarity_matrix
}

# Save the output data to JSON file
with open(OUTPUT_FILE, "w") as f:
    json.dump(output_data, f, indent=4, default=float32_to_float)

print("✅ Analysis complete! Results saved in:", OUTPUT_FILE)

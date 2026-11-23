import fitz  # PyMuPDF for extracting images from PDF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import cv2
import numpy as np

# Load the TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

def preprocess_image(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Increase contrast using adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)

    # Apply binarization (thresholding) to convert to black and white
    _, binarized_img = cv2.threshold(contrast_img, 150, 255, cv2.THRESH_BINARY)

    # Resize to a size that the model expects (e.g., 384x384)
    resized_img = cv2.resize(binarized_img, (384, 384))

    # Convert back to RGB for TrOCR
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    processed_img = Image.fromarray(rgb_img)

    return processed_img

# Function to extract images from a PDF page and preprocess
def extract_and_preprocess_images(pdf_path):
    doc = fitz.open(pdf_path)
    processed_images = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        if pix.width == 0 or pix.height == 0:
            print(f"Warning: Empty image extracted from page {page_num + 1} in {pdf_path}")
            continue

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Preprocess the image
        processed_img = preprocess_image(img)
        processed_images.append(processed_img)

    return processed_images

# Function to process all PDFs in the folder
def process_pdfs_in_folder(pdf_folder, output_folder):
    documents = {}

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {filename}...")

            # Extract and preprocess images from the PDF
            processed_images = extract_and_preprocess_images(pdf_path)

            # Check if images exist
            if not processed_images:
                print(f"Warning: No valid images extracted from {filename}. Skipping.")
                continue

            # Extract text from each processed image using TrOCR
            extracted_text = ""
            for img in processed_images:
                # Convert image to tensor and check shape
                inputs = processor(images=img, return_tensors="pt")
                
                if "pixel_values" not in inputs or inputs["pixel_values"].shape[0] == 0:
                    print(f"Skipping empty image in {filename}.")
                    continue  # Skip empty images
                
                with torch.no_grad():
                    generated_ids = model.generate(inputs["pixel_values"], max_length=1024)

                    if generated_ids.shape[1] == 0:  # Check if output is empty
                        print(f"Warning: Empty text generated for an image in {filename}.")
                        continue

                    text = processor.decode(generated_ids[0], skip_special_tokens=True)
                    extracted_text += text + "\n"

            # Save the extracted text to a file in the output folder
            output_file = os.path.join(output_folder, f"{filename}_extracted.txt")
            with open(output_file, "w") as f:
                f.write(extracted_text)
            documents[filename] = extracted_text

    return documents

# Path to the folder containing your scanned PDFs
PDF_FOLDER = os.path.expanduser("~/Documents/queryai/JFK_Documents_Test")
OUTPUT_FOLDER = os.path.expanduser("~/Documents/queryai/JFK_Documents_Extracted_Text")

# Process PDFs and save extracted text in the output folder
documents = process_pdfs_in_folder(PDF_FOLDER, OUTPUT_FOLDER)

print("Text extraction complete for all PDFs!")

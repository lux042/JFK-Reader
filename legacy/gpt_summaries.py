import openai
import os
import time
import json

# Load API Key from Environment Variable (More Secure)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found! Set it as an environment variable.")

# Function to load extracted text files
def load_extracted_text(folder_path):
    text_files = []
    file_names = []  # To keep track of original file names
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:  # Ignore empty files
                    text_files.append(text)
                    file_names.append(filename)
    return text_files, file_names

# Function to chunk large texts to avoid context length issues
def chunk_text(text, max_tokens=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += 1
        if current_length >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:  # Append last chunk
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to run GPT on the extracted text
def process_with_gpt(text_files, file_names, output_folder):
    responses = []  # List to store responses

    for idx, (text, filename) in enumerate(zip(text_files, file_names)):
        print(f"\nProcessing document {idx + 1}: {filename}")

        # Chunk large text files
        text_chunks = chunk_text(text)
        full_summary = ""

        retries = 0
        max_retries = 5  # Retry up to 5 times if rate-limited

        for chunk_idx, chunk in enumerate(text_chunks):
            prompt = f"Summarize the following text:\n\n{chunk}\n\nSummary:"

            while retries < max_retries:
                try:
                    # Request GPT to generate a summary
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # Upgrade to "gpt-4" if needed
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=500,  # Adjust as needed
                        temperature=0.7
                    )

                    # Extract the generated summary
                    summary = response['choices'][0]['message']['content'].strip()
                    print(f"Chunk {chunk_idx + 1} summary: {summary[:100]}...")  # Preview

                    # Append to full summary
                    full_summary += summary + "\n"

                    break  # Exit retry loop if successful

                except openai.error.RateLimitError:
                    wait_time = (2 ** retries) * 10  # Exponential backoff
                    print(f"Rate limit reached. Retrying in {wait_time} seconds...")
                    retries += 1
                    time.sleep(wait_time)

                except openai.error.OpenAIError as e:
                    print(f"Error processing document {idx + 1}: {e}")
                    break  # Skip this document if an error occurs

        # Save summary to a separate file
        summary_filename = os.path.join(output_folder, f"{filename}_summary.txt")
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(full_summary)

        responses.append({
            'document_name': filename,
            'summary': full_summary
        })

    # Save all responses to a JSON file
    json_output_path = os.path.join(output_folder, "summaries.json")
    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(responses, json_file, indent=4)

    print("\n✅ Summarization complete! Check the output folder.")

# Path to the extracted text files
TEXT_FOLDER = os.path.expanduser("~/Documents/queryai/JFK_Documents_Extracted_Text")
OUTPUT_FOLDER = os.path.expanduser("~/Documents/queryai/JFK_Documents_Summaries")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load text files
text_files, file_names = load_extracted_text(TEXT_FOLDER)

# Process with GPT
process_with_gpt(text_files, file_names, OUTPUT_FOLDER)

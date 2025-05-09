import datetime
import os
import re
import time
import sys
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
from threading import Thread
import itertools
import argparse
from logging.handlers import RotatingFileHandler

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: API key not found. Make sure to set OPENAI_API_KEY in your .env file.")
    exit(1)

# Set the OpenAI API key
client = OpenAI(api_key=API_KEY)

# Logging setup
log_file = "openai_generator.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3),  # 5 MB max file size
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Script started.")

# Loading indicator
stop_loading = False

def loading_indicator():
    """Displays a loading indicator in the terminal."""
    for char in itertools.cycle(['|', '/', '-', '\\']):
        if stop_loading:
            break
        sys.stdout.write(f'\rGenerating response... {char}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rResponse generated!           \n')


def format_as_h1_and_get_title(content):
    """
    Ensures the first line of the content is an h1 in Markdown and extracts the title for the filename.
    
    :param content: The text content returned by the API.
    :return: Tuple (formatted_content, title_for_filename)
    """
    lines = content.splitlines()
    if lines:
        # Extract the first line as the title and clean it
        title_line = lines[0].strip()
        title_line = re.sub(r"\*\*|\*", "", title_line)  # Remove Markdown bold/italic formatting

        # Ensure it's formatted as an h1 tag
        if not title_line.startswith("# "):
            title_line = f"# {title_line}"
        lines[0] = title_line

        # Generate the filename-friendly version of the title
        title_for_filename = re.sub(r"[^a-zA-Z0-9\s-]", "", title_line[2:]).strip().lower()
        title_for_filename = re.sub(r"\s+", "-", title_for_filename)

        formatted_content = "\n".join(lines)
        return formatted_content, title_for_filename
    return content, "untitled"

def fetch_openai_response(prompt, model="gpt-4o"):
    """
    Sends a custom prompt to the OpenAI API and returns the generated text and metadata.
    
    :param prompt: The custom prompt to send.
    :param model: The model to use (default: gpt-4).
    :return: Tuple (text, total_tokens).
    """
    global stop_loading
    try:
        # Start the loading indicator in a separate thread
        loading_thread = Thread(target=loading_indicator)
        loading_thread.start()

        # Record the start time
        start_time = time.time()

        logging.info(f"Sending prompt: {prompt}")

        # Make the API call
        response = client.chat.completions.create(model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Act as a senior AI specialist and professional WordPress developer with over 10 years of experience in integrating and optimizing AI solutions within WordPress environments. Leverage your extensive knowledge of machine learning, deep learning frameworks, WordPress architecture, server configurations, and secure coding standards to provide comprehensive, professional, and actionable content. Ensure your writing is clear, educational, and suitable for a diverse audience—ranging from website administrators and developers to AI enthusiasts. Use Markdown format, starting with an H1 tag, and include IMG_PLACEHOLDER_{number} lines where relevant visuals, charts, or diagrams would enhance understanding. Write in the direct, motivational style and tone of Jocko Willink. "
                    "Avoid using generic phrases like 'In today's digital landscape', 'Now more than ever.', 'WordPress powers over 40 percent of the web, making it a prime target for attackers.', etc. "
                    "Focus on clear, concise, and natural language that resonates with experienced readers. "
                    "Structure the content with clear headings, actionable insights, and practical examples. "
                    "Use Markdown formatting for headings, lists, and code blocks where appropriate."
                )
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=7700)

        # Calculate the total time taken
        total_time = time.time() - start_time
        total_tokens = response.usage.total_tokens

        # Stop the loading indicator
        stop_loading = True
        loading_thread.join()

        logging.info(f"Received response in {total_time:.2f} seconds using {total_tokens} tokens.")
        logging.info(f"Response content: {response.choices[0].message.content}")

        return response.choices[0].message.content, total_time, total_tokens
    except Exception as e:
        stop_loading = True
        logging.error(f"Error during API call: {e}")
        return None, None, None

def save_to_md_file(filename, content):
    """
    Saves the given content to a .md file in the 'content' folder.
    
    :param filename: The name of the file to save the content.
    :param content: The text content to save.
    """
    # Ensure the 'content' folder exists
    os.makedirs("content", exist_ok=True)

    # Save the file in the 'content' folder
    filepath = os.path.join("content", filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info(f"File saved: {filepath}")
    except Exception as e:
        logging.error(f"Failed to save file {filepath}: {e}")

def process_prompts_from_file(file_path):
    """
    Processes multiple prompts from a JSON file and generates Markdown files for each.
    
    :param file_path: Path to the JSON file containing prompts.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            prompts = json.load(file)
        for i, prompt in enumerate(prompts, 1):
            logging.info(f"Processing prompt {i}/{len(prompts)}...")
            response_text, total_time, total_tokens = fetch_openai_response(prompt)
            if response_text:
                formatted_content, title_for_filename = format_as_h1_and_get_title(response_text)
                output_filename = f"{title_for_filename}.md"
                save_to_md_file(output_filename, formatted_content)
            else:
                logging.error(f"Failed to process prompt {i}.")
    except Exception as e:
        logging.error(f"Error processing prompts from file: {e}")

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown files from OpenAI API.")
    parser.add_argument("--prompts", type=str, help="Path to a JSON file containing prompts.")
    args = parser.parse_args()

    if args.prompts:
        process_prompts_from_file(args.prompts)
    else:
        custom_prompt = input("Enter your prompt: ")
        if not custom_prompt.strip():
            logging.error("Prompt cannot be empty.")
            exit(1)

        response_text, total_time, total_tokens = fetch_openai_response(custom_prompt)
        if response_text:
            formatted_content, title_for_filename = format_as_h1_and_get_title(response_text)
            output_filename = f"{title_for_filename}.md"
            save_to_md_file(output_filename, formatted_content)
        else:
            logging.error("Failed to fetch a response.")

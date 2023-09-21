import torch
from pathlib import Path
from PIL import Image
import threading
import rembg
import io
import time
import os

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("Using CPU")

def clean_output_directory(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)
            print(f"Deleted {file_path}")

def remove_background(input_path, output_path, session):
    try:
        with open(input_path, 'rb') as i:
            input_data = i.read()
            output_data = rembg.remove(input_data, session=session)
            output_image = Image.open(io.BytesIO(output_data))  # Convert bytes to PIL Image
            print('saved : ', output_path)
            output_image.save(output_path)
    except Exception as e:
            print(f"Error processing {input_path}: {e}")

def process_image_in_thread(input_file, output_dir, session):
    output_path = str(output_dir / (input_file.stem + ".out.png"))
    remove_background(str(input_file), output_path, session)

def main():
    input_dir = Path('tests/fixtures')
    input_files = list(input_dir.glob('*.jpg'))
    output_dir = input_dir / 'output'

    # Clean the output directory before processing
    clean_output_directory(output_dir)

     # Set the CPU execution provider explicitly
    session = rembg.new_session(execution_providers=['CPUExecutionProvider'])

    # Record start time
    start_time = time.time()

    # One thread per input file
    num_threads = len(input_files)
    threads = []

    for input_file in input_files[:num_threads]:
        thread = threading.Thread(target=process_image_in_thread, args=(input_file, output_dir, session))
        threads.append(thread)
        thread.start()

    print(threading.active_count())
    print(threading.enumerate())

    for thread in threads:
        thread.join()

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()

import torch
from pathlib import Path
from PIL import Image
import threading
import rembg
import io
import time
import os

images_per_thread = 1

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
            # print('saved : ', output_path)
            output_image.save(output_path)
    except Exception as e:
            print(f"Error processing {input_path}: {e}")

def process_image_in_thread(input_files, output_dir, session, start_index, end_index):
    current_thread = threading.current_thread()
    for i in range(start_index, end_index):
        input_file = input_files[i]
        output_path = str(output_dir / (input_file.stem + f"_{current_thread.name}.out.png"))
        print(f"Process in thread {current_thread.name}: {input_file}")
        remove_background(str(input_file), output_path, session)

def main():
    input_dir = Path('100KB')
    input_files = list(input_dir.glob('*.jpg'))
    output_dir = input_dir / 'output'

    # Clean the output directory before processing
    clean_output_directory(output_dir)

     # Set the CPU execution provider explicitly
    session = rembg.new_session(execution_providers=['CPUExecutionProvider'])

    # Record start time
    start_time = time.time()

    num_threads = 8
    threads = []

    # Split input files into equal parts for each threads
    batch_size = len(input_files) // num_threads
    # input_files_batches = [input_files[i:i + batch_size] for i in range(0, len(input_files), batch_size)]

    for i in range(num_threads):
        # Executing same function but input files are splited into different batches and run in the created threads
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(input_files))
        thread = threading.Thread(target=process_image_in_thread, args=(input_files, output_dir, session, start_index, end_index))
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

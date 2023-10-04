import torch
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import io
import time
import os
import concurrent.futures

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")


def clean_output_directory(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)
    print("Cleared Output Directory")
    print("=====================================")


def process_image(input_file, output_dir, session, input_files):
    try:
        input_path = str(input_file)
        output_path = str(output_dir / (input_file.stem + ".out.png"))

        with open(input_path, 'rb') as i:
            input_data = i.read()

            # Remove the background from the input image using CUDA
            output_data = remove(input_data, session=session)

            # Convert the output data to a PIL Image
            output_image = Image.open(io.BytesIO(output_data))

            # Save the output image
            output_image.save(output_path)

        # Print the results
        print(f"Processed {input_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def process_images_on_gpu(input_files, output_dir, thread_num):
    # Create a new session for GPU processing
    providers = ['CUDAExecutionProvider']
    session = new_session(providers=providers)

    # Clean the output directory before processing
    clean_output_directory(output_dir)

    start_time = time.time()  # Record start time

    # Process each input image on the GPU in parallel
    num_images = len(input_files)
    print(f"{num_images} images detected")
    print("Processing images...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = []
        for input_file in input_files:
            future = executor.submit(process_image, input_file, output_dir, session, input_files)
            futures.append(future)

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time
    print("=====================================")
    print(f"GPU Execution time: {execution_time:.2f} seconds")


def main():
    input_dir = Path('100KB-50photo')
    input_files = list(input_dir.glob('*.jpg'))
    output_dir = input_dir / 'output'

    # Sort the input files
    input_files.sort()

    process_images_on_gpu(input_files, output_dir, thread_num=8)


if __name__ == "__main__":
    main()

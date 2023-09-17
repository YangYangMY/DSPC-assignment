import torch
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
import io
import time
import os

def clean_output_directory(output_dir):
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)
            print(f"Deleted {file_path}")

def ProcessImage(use_gpu):
    input_dir = Path('tests/fixtures')
    input_files = list(input_dir.glob('*.jpg'))
    output_dir = input_dir / 'output'

    # Clean the output directory before processing
    clean_output_directory(output_dir)

    gpu_execution_time = 0.0  # Initialize GPU execution time
    cpu_execution_time = 0.0  # Initialize CPU execution time

    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        torch.cuda.synchronize()  # Ensure that previous GPU operations are finished
        # Set the CUDA execution provider explicitly
        session = new_session(execution_providers=['CUDAExecutionProvider'])
    else:
        print("Using CPU")
        # Set the CUDA execution provider explicitly
        session = new_session(execution_providers=['CPUExecutionProvider'])

    for input_file in input_files:
        try:
            input_path = str(input_file)
            output_path = str(output_dir / (input_file.stem + ".out.png"))

            with open(input_path, 'rb') as i:
                input_data = i.read()

                if use_gpu and torch.cuda.is_available():
                    gpu_start_time = time.time()  # Record GPU start time
                    output_data = remove(input_data, session=session)
                    output_image = Image.open(io.BytesIO(output_data))  # Convert bytes to PIL Image
                    print('saved : ', output_path)
                    output_image.save(output_path)
                    torch.cuda.synchronize()  # Ensure that all GPU operations are finished
                    gpu_end_time = time.time()  # Record GPU end time
                    gpu_execution_time += (gpu_end_time - gpu_start_time)
                else:
                    cpu_start_time = time.time()  # Record CPU start time
                    output_data = remove(input_data, session=session)
                    output_image = Image.open(io.BytesIO(output_data))  # Convert bytes to PIL Image
                    print('saved : ', output_path)
                    output_image.save(output_path)
                    cpu_end_time = time.time()  # Record CPU end time
                    cpu_execution_time += (cpu_end_time - cpu_start_time)  # Measuring CPU time here

        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    if use_gpu and torch.cuda.is_available():
        print(f"GPU Execution time: {gpu_execution_time:.2f} seconds")
    else:
        print(f"CPU Execution time (without GPU): {cpu_execution_time:.2f} seconds")

def main():
    # Run the code first with CPU
    ProcessImage(use_gpu=False)

    # Then run the code with GPU
    ProcessImage(use_gpu=True)

if __name__ == "__main__":
    main()

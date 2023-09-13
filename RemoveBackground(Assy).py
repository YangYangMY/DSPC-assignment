import torch
from pathlib import Path
from rembg import remove, new_session
from PIL import Image
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


def ProcessImage():
    input_dir = Path('tests/fixtures')
    input_files = list(input_dir.glob('*.jpg'))
    output_dir = input_dir / 'output'

    # Clean the output directory before processing
    clean_output_directory(output_dir)

    session = new_session()

    start_time = time.time()  # Record start time

    for input_file in input_files:
        try:
            input_path = str(input_file)
            output_path = str(output_dir / (input_file.stem + ".out.png"))

            with open(input_path, 'rb') as i:
                input_data = i.read()

                output_data = remove(input_data, session=session)
                output_image = Image.open(io.BytesIO(output_data))  # Convert bytes to PIL Image
                print('saved : ', output_path)
                output_image.save(output_path)

        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


def main():
    ProcessImage()


if __name__ == "__main__":
    main()

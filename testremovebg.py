from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rembg import remove, new_session

def process_image(file):
    session = new_session()
    input_path = str(file)
    output_path = str(file.parent / 'output' / (file.stem + ".out.png"))

    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            print('hi')
            input_data = i.read()
            output_data = remove(input_data, session=session)
            o.write(output_data)

def main():

    input_files = list(Path('tests/fixtures').glob('*.jpg'))

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_image, input_files)

if __name__ == "__main__":
    main()


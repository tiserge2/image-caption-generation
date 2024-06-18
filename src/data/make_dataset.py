import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
from PIL import Image
import io
import ast  # this is 
import json
from tqdm import tqdm
import os
import argparse

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


def convert_bytes_to_Image(byte_array):
    """
    Convert a byte array into an image.

    Args:
        byte_array (str): Byte array representing the image.

    Returns:
        Image: Image object created from the byte array.
    """
    return Image.open(io.BytesIO(ast.literal_eval(byte_array)['bytes']))


def txt_clean(caption):
    """
    Clean a caption by converting to lowercase, removing punctuation, and removing words containing numbers.

    Args:
        caption (str): The original caption text.

    Returns:
        str: The cleaned caption text.
    """
    table = str.maketrans('', '', string.punctuation)
    caption = caption.replace("-", " ")
    descp = caption.split()
    # Uppercase to lowercase
    descp = [wrd.lower() for wrd in descp]
    # Remove punctuation from each token
    descp = [wrd.translate(table) for wrd in descp]
    # Remove hanging 's and a
    descp = [wrd for wrd in descp if (len(wrd) > 1) or wrd == 'a']
    # Remove words containing numbers
    descp = [wrd for wrd in descp if wrd.isalpha()]
    # Converting back to string
    caption = ' '.join(descp)
    return caption


def handle_special_word(data_captions, longest_seq):
    """
    Add special tokens to captions and pad them to the longest sequence length.

    Args:
        data_captions (list): List of data captions.
        longest_seq (int): Length of the longest caption sequence.

    Returns:
        list: List of data captions with special tokens and padding.
    """
    pad_seq = " <pad>"
    for j, data in enumerate(data_captions):
        for i, caption in enumerate(data_captions[j]['captions']):
            length_capt = len(caption.split(" "))
            diffe = longest_seq - length_capt
            
            if diffe > 0:
                data_captions[j]['captions'][i] = '<start> ' + caption + ' <eos>' + pad_seq * diffe
            elif diffe == 0:
                data_captions[j]['captions'][i] = '<start> ' + caption + ' <eos>'
                
    return data_captions


def build_interim(data_path, test_data, build_test, shortage, examples, data_caption_name):
    """
    Build an interim dataset from raw data, cleaning captions and saving images.

    Args:
        data_path (str): Path to the raw data directory.
        test_data (list): List of filenames for test data.
        build_test (bool): Whether to build a test dataset.
        shortage (bool): Whether to limit the number of examples.
        examples (int): Number of examples to include if shortage is True.
        data_caption_name (str): Base name for the output JSON file.

    Returns:
        None
    """
    data_captions = []
    longest_seq = 0
    i = 0
    missing = 0

    for dirname, _, filenames in os.walk(os.path.join(data_path, "raw")):
        for filename in tqdm(filenames):
            if filename.split(".")[-1] == "csv":
                df = pd.read_csv(os.path.join(dirname, filename))
                print(f"len df: {len(df)}")
                for index, row in df.iterrows():
                    filename = row['filename'].split("/")[-1]
                    if (not build_test and filename in test_data) or (build_test and filename not in test_data):
                        missing += 1
                        continue

                    cleaned_captions = [txt_clean(x) for x in ast.literal_eval(row['captions'])[0].strip().split(".") if x != ""]
                    local_long_seq = max([len(x.split(" ")) for x in cleaned_captions])
                    longest_seq = local_long_seq if local_long_seq > longest_seq else longest_seq
                    
                    captions = [txt_clean(x) for x in cleaned_captions]
                    if len(captions) != 5:
                        captions.append(captions[3])
                    
                    image = convert_bytes_to_Image(row['image'])
                    data_captions.append({
                        "filename": filename,
                        "captions": captions
                    })

                    # Save file to disk if it doesn't exist
                    image_path = os.path.join(data_path, "interim/images", filename)
                    if not os.path.exists(image_path):
                        image.save(image_path, format="jpeg")

                    i += 1
                    if not build_test:
                        if i == examples and shortage:
                            break
                if not build_test:
                    if i == examples and shortage:
                        break
    
    if build_test:
        data_caption_name = data_caption_name + "_test.json" 
    else:
        if len(test_data) == 0:
            data_caption_name = data_caption_name + "_all.json"
        else:
            if not shortage:
                data_caption_name = data_caption_name + "_train.json"
            else:
                data_caption_name = data_caption_name + f"_{examples}_train.json"
    
    print(f"missing: {missing}")
    data_captions = handle_special_word(data_captions, longest_seq)
    data_captions_path = os.path.join(data_path, "interim", data_caption_name)
    with open(data_captions_path, "w") as f:
        json.dump(data_captions, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--data_path', type=str, default=r"/home/sagemaker-user/rscid/data",
                        help='Path to the data directory')
    parser.add_argument('--test_data_path', type=str, default=r"/home/sagemaker-user/rscid/data/interim/test_data.json",
                        help='Path to the test data JSON file')
    parser.add_argument('--build_test', action='store_true',
                        help='Whether to build test data')
    parser.add_argument('--shortage', action='store_true',
                        help='Whether to consider shortage')
    parser.add_argument('--build_all', action='store_true',
                        help='Whether to build all data')
    parser.add_argument('--examples', type=int, default=1000,
                        help='Number of examples')
    parser.add_argument('--data_caption_name', type=str, default="captions",
                        help='Name of the data caption')

    args = parser.parse_args()

    with open(args.test_data_path, "r") as f:
        test_data = json.load(f)

    if args.build_all:
        test_data = []

    build_interim(args.data_path, test_data, args.build_test, args.shortage, args.examples, args.data_caption_name)

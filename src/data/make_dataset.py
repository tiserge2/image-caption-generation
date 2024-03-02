import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string

from PIL import Image
import io
import ast  # this is 
import json
from tqdm import tqdm
import os


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def convert_bytes_to_Image(byte_array) : 
    """
        Convert byte array into image to be saved later
    """
    return Image.open(io.BytesIO((ast.literal_eval(byte_array)['bytes'])))

#Data cleaning function will convert all upper case alphabets to lowercase, removing punctuations and words containing numbers
def txt_clean(caption):
    table = str.maketrans('','',string.punctuation)
    caption.replace("-"," ")
    descp = caption.split()
    #uppercase to lowercase
    descp = [wrd.lower() for wrd in descp]
    #remove punctuation from each token
    descp = [wrd.translate(table) for wrd in descp]
    #remove hanging 's and a
    descp = [wrd for wrd in descp if(len(wrd)>1)]
    #remove words containing numbers with them
    descp = [wrd for wrd in descp if(wrd.isalpha())]
    #converting back to string
    caption = ' '.join(descp)
    return caption

def build_interim():
    data_captions = []
    i = 0
    for dirname, _, filenames in os.walk("./data/raw/"):
        for filename in tqdm(filenames):
            if filename.split(".")[-1] == "csv":
                df = pd.read_csv(os.path.join(dirname, filename))
                for index, row in df.iterrows():
                    filename = row['filename'].split("/")[-1]
                    captions = [txt_clean(x) for x in ast.literal_eval(row['captions'])[0].strip().split(".") if x != ""]
                    image = convert_bytes_to_Image(row['image'])

                    data_captions.append({
                        "filename": filename,
                        "captions": captions
                    })

                    # save file to disk if it doesn't exist
                    # image_path = os.path.join("../data/interim/images", filename)
                    # if not os.path.exists(image_path):
                    #     image.save(image_path, format="jpeg")
                    i += 1
                    if i == 2:
                        break

    data_captions_path = os.path.join("./data/interim/", "caption_2.json")
    with open(data_captions_path, "w") as f:
        json.dump(data_captions, f, indent=4)


if __name__ == '__main__':
    build_interim()

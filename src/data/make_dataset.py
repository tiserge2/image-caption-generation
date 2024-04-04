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


def build_interim(data_path, test_data, build_test, shortage, examples, data_caption_name):
    data_captions = []
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
                    captions = ["<start> " + txt_clean(x) + " <eos>" for x in ast.literal_eval(row['captions'])[0].strip().split(".") if x != ""]
                    # captions = [x for x in ast.literal_eval(row['captions'])[0].strip().split(".") if x != ""]
                    image = convert_bytes_to_Image(row['image'])

                    data_captions.append({
                        "filename": filename,
                        "captions": captions
                    })

                    # save file to disk if it doesn't exist
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


    data_captions_path = os.path.join(data_path, "interim", data_caption_name)
    with open(data_captions_path, "w") as f:
        json.dump(data_captions, f, indent=4)


if __name__ == '__main__':
    data_path_ = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data"
    test_data = r"/Users/sergiosuzerainosson/Documents/project/universite_project/s4/modelisation_vision/image-caption-generation/data/interim/test_data.json"
    with open(test_data, "r") as f:
        test_data = json.load(f)
    build_test = False
    shortage = True
    examples = 10
    data_caption_name = "captions"
    build_interim(data_path_, test_data, build_test, shortage, examples, data_caption_name)

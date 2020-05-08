
from utils import load_dataframe, write_json, load_json
import numpy as np
from PIL import Image
import hashlib
import time
from tqdm import tqdm
import os
import random


# preprocess covid images
def get_covid_images(covid_dataset_path: str="", covid_metadata_path: str="", shape: tuple=(), save_to: str="", label: int=0) -> int:
    metadata = load_dataframe(file_=covid_metadata_path)

    count = 0
    for (i, row) in tqdm(metadata.iterrows(), ncols=150):
        if row["finding"] == "COVID-19" and row["view"] == "PA":
            count += 1

            covid_image_file = covid_dataset_path + row["filename"].split(os.path.sep)[-1]
            new_image_file = save_to + str(label) + "_" + str(hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest())[:16] + "." + covid_image_file.split(".")[-1]
            
            image = Image.open(covid_image_file)
            image = image.resize(shape)
            image.save(new_image_file)
    
    return count


# preprocess healthy and other-type pneumonia images
def get_images(dataset_path: str="", shape: tuple=(), save_to: str="", label: int=0, max_amount=100):
    image_names = os.listdir(dataset_path)[:max_amount]

    for image_name in tqdm(image_names, ncols=150):
        image_file = dataset_path + image_name
        new_image_file = save_to + str(label) + "_" + str(hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest())[:16] + "." + image_file.split(".")[-1]

        image = Image.open(image_file)
        image = image.resize(shape)
        image.save(new_image_file)


# count classes
def class_count(dataset: str=""):
    a, b, c = 0, 0, 0

    for n in os.listdir("datasets/final-dataset/"):
        if n.split("_")[0] == "0":
            a += 1
        elif n.split("_")[0] == "1":
            b += 1
        elif n.split("_")[0] == "2":
            c += 1

    print("healthy", a, ", pneumonia", b, ", covid", c)


if __name__ == "__main__":
    labels = {"healthy": 0, "pneumonia": 1, "covid": 2}
    shape = (512, 512)

    final_folder = "datasets/final-dataset/"

    covid_dataset_path = "datasets/covid-chestxray-dataset/images/"
    covid_metadata_path = "datasets/covid-chestxray-dataset/metadata.csv"

    pneumonia_dataset_path = "datasets/pneumonia-chestxray-dataset/"

    healthy_dataset_path = "datasets/healthy-chestxray-dataset/"

    # remove old data
    print("removing previous images")
    os.system("rm " + final_folder + "*")

    # preprocess covid images
    print("\npreprocessing covid xray images")
    amount = get_covid_images(covid_dataset_path=covid_dataset_path, covid_metadata_path=covid_metadata_path, shape=shape, save_to=final_folder, label=labels["covid"])

    # the will always be less covid-xray images than normal, therefore: the amount of covid images is the maximum amount of images per class
    max_amount = amount

    # preprocess other-type pneumonia images
    print("\npreprocessing other-type pneumonia images")
    get_images(dataset_path=pneumonia_dataset_path, shape=shape, save_to=final_folder, label=labels["pneumonia"], max_amount=max_amount)

    # preprocess healthy images
    print("\npreprocessing healthy xray images")
    get_images(dataset_path=healthy_dataset_path, shape=shape, save_to=final_folder, label=labels["healthy"], max_amount=max_amount)

    # print amount of each class (should all be the same)
    class_count(dataset=final_folder)


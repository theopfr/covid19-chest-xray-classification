
import pandas as pd
import json
import os
from termcolor import colored


""" read from dataframe """
def load_dataframe(file_: str=""):
    return pd.read_csv(file_)

""" save to dataframe """
def write_dataframe(content, file_: str=""):
    content.to_csv(file_)

""" read from json """
def load_json(file_: str=""):
    with open(file_, "r") as f:
        return json.load(f)

""" save to json """
def write_json(content: dict, file_: str=""):
    with open(file_, "w") as f:
        json.dump(content, f)

""" copy files to folder """
def copy_to_folder(files: list=[], src: str="", dest: str=""):
    for file_name in files:
        os.system("cp -i " + src + file_name + " " + dest)

""" prints out training progress """
def show_progress(epochs, epoch, loss, val_accuracy, val_loss):
    epochs = colored(epoch, "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    loss = colored(round(loss, 6), "cyan", attrs=['bold'])
    accuracy = colored(round(val_accuracy, 4), "cyan", attrs=['bold']) + colored("%", "cyan", attrs=['bold'])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=['bold'])

    print("epoch {} - loss: {} - val_acc: {} - val_loss: {}".format(epochs, loss, accuracy, val_loss), "\n")

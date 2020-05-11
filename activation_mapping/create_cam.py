import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from PIL import Image
import skimage.transform

from cam_model import Model
from run_model import RunModel



def train_cam_model(model_file: str="", evaluation_treshold: float=0.75):
    runModel = RunModel(
        Model=Model,
        model_file=model_file, 
        dataset_path="../datasets/final-dataset/",
        test_size=0.1,
        val_size=0.1,
        epochs=15,
        batch_size=16,
        lr=0.0001,
        dropout_chance=0.4,
        lr_decay=0.1,
        evaluation_treshold=evaluation_treshold)

    #runModel.train()
    runModel.test()




# create class-activation-map of 
class CAM:
    def __init__(self, Model, model_file: str="", image_file: str="", shape: tuple=()):
        self.model = Model().cuda()
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        
        self.image = np.array(Image.open(image_file))
        self.shape = shape

    def _preprocess_image(self, image):
        c, w, h = self.shape[0], self.shape[1], self.shape[2]
        try:
            image = image[:, :, 0]
        except:
            pass
        image = image / 255
        image = torch.Tensor(image).reshape(1, c, w, h).cuda()

        return image

    def _get_class_index(self, prediction, threshold: float=0.75) -> int:
        # create one-hot-encoding
        prediction = [1 if e >= threshold else 0 for e in prediction]

        # check if its a valid output (one 1 and the rest 0s)
        one = []
        for i in range(len(prediction)):
            if prediction[i] == 1:
                one.append(i)
        
        if 0 < len(one) < 2:
            return one[0]
        else:
            return -1


    def create_class_acitvation_map(self):
        preprocessed_image = self._preprocess_image(self.image)
        prediction, feature_maps = self.model(preprocessed_image, return_fmaps=True)
        prediction = prediction.cpu().detach().numpy()[0]

        class_idx = self._get_class_index(prediction, threshold=0.75)
        print(class_idx)
        if class_idx == -1:
            raise Exception("Unvalid output:", prediction, ". No class predicted.")
        
        weights = self.model.dense1.weight[class_idx]
        weights = weights.cpu().detach().numpy()
        feature_maps = feature_maps.cpu().detach().numpy().reshape(512, 9, 9)

        cam = np.dot(feature_maps.T, weights)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        cam = skimage.transform.resize(cam, (512, 512))

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.image, cmap="gray")
        axs[1].imshow(self.image, cmap="gray")
        axs[1].imshow(cam, cmap="jet", alpha=0.4)

        plt.show()



if __name__ == "__main__":
    model_file = "model.pt"
    
    """ train cam-compatible model (comment if already trained) """
    #train_cam_model(model_file=model_file, evaluation_treshold=0.75)

    """ plot cam """
    # datasets/final-dataset/2_d258359cb0576178.jpg
    # 
    cam = CAM(Model, model_file=model_file, image_file="../datasets/final-dataset/2_ddb4f7f4f4ff83d9.jpeg", shape=(1, 512, 512))
    cam.create_class_acitvation_map()


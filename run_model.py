
from utils import show_progress
from test_metrics import validate_accuracy, precision, recall, plot
from model import Model
from chestXrayDataset import ChestXrayDataset

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms


classes = {"healthy": [1, 0, 0], "penumonia": [0, 1, 0], "covid": [0, 0, 1]}


class RunModel:
    def __init__(self, model_file: str="", dataset_path: str="", test_size: float=0.1, val_size: float=0.1, batch_size: int=64, epochs: int=100, lr: float=1e2, dropout_chance: float=0.5, lr_decay: float=0.1):
        self.model_file = model_file

        self.dataset_path = dataset_path
        self.test_size = test_size
        self.val_size = val_size

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_chance = dropout_chance
        self.lr_decay = lr_decay

        self.train_set, self.validation_set, self.test_set = self._create_dataloader()

    """ creates dataloader """
    def _create_dataloader(self):
        dataset = ChestXrayDataset(self.dataset_path)

        test_amount, val_amount = int(dataset.__len__() * self.test_size), int(dataset.__len__() * self.val_size)

        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
            (dataset.__len__() - (test_amount + val_amount)), 
            test_amount, 
            val_amount
        ])

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=True,
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.batch_size / 2),
            num_workers=1,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.batch_size / 2),
            num_workers=1,
            shuffle=True,
        )

        return train_dataloader, val_dataloader, test_dataloader

    """ calculates accuracy """
    def _validate(self, model, dataset, threshold: float=0.75):
        validation_dataset = self.validation_set
        model = model.eval()

        total_targets, total_predictions = [], []

        for images, targets in tqdm(validation_dataset, desc="validating", ncols=150):
            images = images.float().cuda()
            targets = targets.float().cuda()

            predictions = model(images)

            for i in range(predictions.size()[0]):
                total_targets.append(targets[i].cpu().detach().numpy())
                total_predictions.append(predictions[i].cpu().detach().numpy())

        # calculate accuracy
        accuracy = validate_accuracy(total_targets, total_predictions, threshold=threshold)

        # calculate loss
        criterion = nn.MSELoss()
        loss = criterion(predictions, targets).item()

        # calculate precision of all labels
        precisions = precision(total_targets, total_predictions, classes=list(classes.values()), threshold=threshold)

        # calculate recall of all labels
        recalls = recall(total_targets, total_predictions, classes=list(classes.values()), threshold=threshold)

        return accuracy, loss, precisions, recalls

    """ trains model """
    def train(self, continue_: bool=False):        
        training_dataset = self.train_set

        model = Model(dropout_chance=self.dropout_chance).cuda()

        if continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        loss_data, validation_accuracy_data, validation_loss_data, precision_data, recall_data = [], [], [], [], []
        for epoch in range(1, self.epochs+1):

            epoch_loss = []
            for images, targets in tqdm(training_dataset, desc="epoch", ncols=150):
                optimizer.zero_grad()

                images = images.float().cuda()
                targets = targets.float().cuda()

                predictions = model.train()(images)

                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            current_loss = np.mean(epoch_loss)
            current_val_accuracy, current_val_loss, precisions, recalls = self._validate(model, self.validation_set, threshold=0.75)

            show_progress(self.epochs, epoch, current_loss, current_val_accuracy, current_val_loss)

            loss_data.append(current_loss)
            validation_accuracy_data.append(current_val_accuracy)
            validation_loss_data.append(current_val_loss)
            precision_data.append(precisions)
            recall_data.append(recalls)

            torch.save(model.state_dict(), self.model_file)

        print("\n finished training")
        precision_data = list(zip(*precision_data))
        recall_data = list(zip(*recall_data))
        plot(loss_data, validation_accuracy_data, validation_loss_data, precision_data, recall_data, classes=list(classes.keys()))

    """ tests dataset """
    def test(self, show_examples: bool=True):
        # load dataset
        testing_dataset = self.test_set

        # load model
        model = Model(dropout_chance=0.0)
        model.load_state_dict(torch.load(self.model_file))
        model.cuda().eval()

        # calculate accuracy of the trained model
        accuracy, _, precisions, recalls = self._validate(model, self.test_set, threshold=0.75)
        print("test accuracy:", round(accuracy, 4), "%")
        print("precisions:", precisions)
        print("recalls:", recalls)

        # show examples
        if show_examples:
            for images, targets in testing_dataset:
                images = images.float().cuda()
                targets = targets.float().cuda()
                outputs = model(images)

                for idx in range(images.size()[0]):
                    image, target, output = images[idx], targets[idx], outputs[idx]

                    print("\nexpected: ", target.cpu().detach().numpy(), "\ngot:      ", np.around(output.cpu().detach().numpy()))
                    print("\n_______________________\n")

                    example = image.cpu().detach().numpy().reshape(512, 512, 1)
                    example = example * 255
                    
                    plt.imshow(image.cpu().detach().numpy().reshape(512, 512, 1))
                    plt.title("got: " + str(output.cpu().detach().numpy()))
                    plt.show()



if __name__ == "__main__":
    runModel = RunModel(
        model_file="models/model_1.pt", 
        dataset_path="datasets/final-dataset/",
        test_size=0.075,
        val_size=0.075,
        epochs=10,
        batch_size=16,
        lr=0.0001,
        dropout_chance=0.4,
        lr_decay=0.1)

    runModel.train(continue_=False)
    runModel.test(show_examples=True)



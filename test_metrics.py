
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from PIL import Image


# check how much percent were classified correctly
def validate_accuracy(y_true, y_pred, threshold: float=0.75) -> int:
    correct_in_batch = 0
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]
        output = [1 if e >= threshold else 0 for e in output]

        if list(target) == output:
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


# calculate precision scores of all labels (against all labels)
def precision(y_true, y_pred, classes=[[1, 0], [0, 1]], threshold: float=0.75) -> float:

    """ FIX THIS SHIT """

    total_prediction_of_classes, total_true_prediction_of_classes = [0 for i in range(len(classes))], [0 for i in range(len(classes))]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]
        output = [1 if e >= threshold else 0 for e in output]

        for j in range(len(classes)):
            if output == classes[j]:
                total_prediction_of_classes[j] += 1

                if output == list(target):
                    total_true_prediction_of_classes[j] += 1

    all_precisions = [0 for i in range(len(classes))]
    for i in range(len(classes)):
        if total_prediction_of_classes[i] > 0:
            all_precisions[i] = round((total_true_prediction_of_classes[i] / total_prediction_of_classes[i]), 5)
        else:
            all_precisions[i] = 0

    return all_precisions


# calculate recall scores of all labels (against all labels)
def recall(y_true, y_pred, classes=[[1, 0], [0, 1]], threshold: float=0.75) -> float:
    total_prediction_of_classes, total_true_of_classes = [0 for i in range(len(classes))], [0 for i in range(len(classes))]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]
        output = [1 if e >= threshold else 0 for e in output]

        for j in range(len(classes)):
            if output == classes[j]:
                total_prediction_of_classes[j] += 1

            if list(target) == classes[j]:
                total_true_of_classes[j] += 1

    all_recalls = [0 for i in range(len(classes))]
    for i in range(len(classes)):
        if total_true_of_classes[i] > 0:
            all_recalls[i] = round((total_prediction_of_classes[i] / total_true_of_classes[i]), 5)
        else:
            all_recalls[i] = 0

    return all_recalls


# create two subplots for loss and accuracy history
def plot(train_loss: list, val_acc: list, val_loss: list, precisions: list, recalls: list, classes: list=[""], save_to: str=""):
    plt.style.use("ggplot")

    fig, axs = plt.subplots(3)
    fig.set_size_inches(7, 8)

    xs = range(len(train_loss))

    fig = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    axs1 = fig.add_subplot(spec[0, 0])
    axs2 = fig.add_subplot(spec[1, 0])
    axs3 = fig.add_subplot(spec[0, 1])
    axs4 = fig.add_subplot(spec[1, 1])

    # plot validation loss and accuracy
    loss_p = axs1.plot(xs, val_loss, "r", label="val-loss")
    acc_p = axs1.plot(xs, train_loss, "b", label="train-loss")
    axs1.legend()
    axs1.set_title("val-/ train-loss")

    # plot train loss
    axs2.plot(xs, val_acc, "r")
    axs2.set_title("val-acc")

    colors = ["b", "r", "g", "y", "orange", "purple", "brown"]
    np.random.shuffle(colors)

    # plot recalls
    for i in range(len(recalls)):
        axs3.plot(xs, recalls[i], colors[i], label=classes[i])

    axs3.legend()
    axs3.set_title("recall of every class")

    # plot precisions
    for i in range(len(precisions)):
        axs4.plot(xs, precisions[i], colors[i], label=classes[i])

    axs4.legend()
    axs4.set_title("precision of every class")

    plt.savefig(save_to)
    plt.show()


# visualize feature maps
def visualize_feature_maps(Model, model_file: str="", image_file: str="", shape: tuple=()):
    model = Model().cuda()
    model.load_state_dict(torch.load(model_file))
    model = model.eval()

    c, w, h = shape[0], shape[1], shape[2]

    image = np.array(Image.open(image_file))
    image = image[:, :, 0]
    image = image / 255
    image = torch.Tensor(image).reshape(1, c, w, h).cuda()

    _ = model(image, print_=True, visualize=True)




# Due to the small dataset (145 samples per class), the test results can not be considered accurate!


# | dataset:

### The dataset is assembled of [this](https://github.com/ieee8023/covid-chestxray-dataset) dataset for covid images (PA xray type only) and [this](https://github.com/ieee8023/covid-chestxray-dataset) dataset for healthy and other-type pneumonia images

### images (1, 512, 512):
![samples](plots/sample.jpeg)
##### (left healthy, right covid sample)

# | model:

## architecture:
![architecture](plots/architecture.png)


# | training:

#### - epochs: 12
#### - batch-size: 16 (limited amount because of image size and GPU resources)
#### - learning-rate: 0.0001
#### - dropout (dense-layers): 40%

## trainings-progress:
### test-accuracy: 93.0233%
##### (high precision and recall, probably because of the small dataset)
### recalls:
#### covid: 100%, other-type pneumonia: 86.667%, healthy: 93.75%
### precisions:
#### covid: 100%, other-type pneumonia: 100%, healthy: 100%
![training](plots/training_2.png)

## example outputs:
![output_samples](plots/output_sample.jpeg)
##### (correct classified: left healthy, right covid sample)


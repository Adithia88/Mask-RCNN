# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample](assets/1.PNG)

The repository includes:
* convert json to COCO
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset


The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). If you work on 3D vision, you might find our recently released [Matterport3D](https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/) dataset useful as well.
This dataset was created from 3D-reconstructed spaces captured by our customers who agreed to make them publicly available for academic use. You can see more examples [here](https://matterport.com/gallery/).


## Requirements
Python 3.6, TensorFlow 2.0, and other common packages listed in `requirements.txt`.

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset](http://cocodataset.org/#home)
* Download the 5K [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0)
  and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)
  subsets. More details in the original [Faster R-CNN implementation](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).

If you use Docker, the code has been verified to work on
[this Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/).


## Installation
1. Clone this repository
2. Download this file and put the file like picture below (https://drive.google.com/file/d/1QMlACi1SQDaDQ8Sl9cTXfRwQQyf9k6xC/view?usp=sharing) and ( https://drive.google.com/file/d/1lOJD620DM3IFgBPPbHrCbTDHjWqlCXHy/view?usp=sharing)

![Put file like this](assets/2.PNG)

3. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
4. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
5. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
6. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)
    
    
This is an implementation of Plant instance segmentation using deep learning Mask-RCNN on Python 3. This code generates masking and objecdetectiont in every objects in one image.


# Getting Started
* open sample Load_model R50_E08.ipynb in the main page  Is the easiest way to start. It shows an example of plant instance segmentation (testing only)
* convert json to COCO
* open train_tunning.ipynb in the main page, this codes for training (training)
* open Evaluation_Gen_IoU.ipynb, test_mask_rcnn_map.ipynb and test_mask_rcnn_dice_coef_test_data.py for evaluation (evaluation)


# step to convert json to COCO 

## 1. Prepare dataset and make sure the path 
open code labelme2coco.py in dataset/custom  and  put dataset (dataset/custom) too in your main path. Make sure u put the data and codes like my picture below!
like u see i divide my data to 3 part train, val and test (just ignore the .json that the result when u convert it)

![](assets/3.PNG)

## 2.convert json to COCO 
just change the name folder what u want convert , so in this case we convert 3 times (train,val and test) 

![](assets/4.PNG)

## 3. Execute program (Labelme2coco.py) 
press F5.


# Step to train  with your own data

## 1. train own dataset
This example will explain which part u must change to train your own dataset. open code with name train_tuning.ipynb  or train_tuning.py in sample folder (Ipynb for jupyter , Py for spyder)  in this tutorial im gonna use jupyter notebook (.ipynb)

the line codes u must change 

![](assets/5.PNG)

Root_dir = (path) which u put whole files 

![](assets/6.PNG)

NUM_classes = 1 + (your own data class)* 
BACKBONE = 'resnet50'  (u can use 2 backbone here resnet50 or resnet101)

![](assets/7.PNG)

dataset_train.load_data(Root_dir + your data train path )
dataset_val.load_data(Root_dir + your data val path )

![](assets/8.PNG)

epoch =  u can config the epoch i make it 25 for heads

![](assets/9.PNG)

epoch =  2x heads epochs (!warning just follow what i said)

![](assets/10.PNG)

so this is bonus stage u can load directly yours model in 1 code  
real_test_dir = Root_dir + (your test data path) 

## 3. Execute all program
press shift+enter per line or Run all 



# Step to testing (load model) 

## 1. prepare own model (.h5)
Make sure u put the models and know the model name, picture below

![](assets/11.PNG)

so for my model i just have my last models in logs/plant folder. My model name is mask_rcnn_custom_0050.h5

## 2. test own dataset
open load_model R50_E08.ipynb code in samples folder, then make sure the path is right and name of the model too.

![](assets/5.PNG)

Root_dir = (path) which u put whole files  (same as train)

![](assets/7.PNG)

dataset_train.load_data(Root_dir + your data train path )  (same as train)
dataset_val.load_data(Root_dir + your data val path )      (same as train)

![](assets/11.PNG)

model_path = Root_dir + (your own models)  

![](assets/10.PNG)
real_test_dir = Root_dir + (your test data path)    (same as train)

## 3. Execute all program
press shift+enter per line or Run all 



# Step to evaluate the model

## 1. evaluate model
so i make some evaluate you can just try and make the same path too like load model
evaluate include :
*IoU
*Dice
*mAP

## 2.Execute program
press shift+enter per line or Run all 



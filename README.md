# CueCAn: Cue Driven Contextual Attention Unit for Identifying Missing Traffic Signs on Uncontrained Roads
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ujnqXKmQHUQP91COmuu_HXkjitzh0fGC?usp=sharing)

## Dataset Download
Dataset Name: <i>Missing Traffic Signs Video Dataset (MTSVD)</i>.

Available for download at the following link:
https://idd.insaan.iiit.ac.in/dataset/download/

## Conference Proceeding

International Conference on Robotics and Automation 2023
https://ieeexplore.ieee.org/document/10161576 

## Inference in Colab

https://colab.research.google.com/drive/1ujnqXKmQHUQP91COmuu_HXkjitzh0fGC?usp=sharing

### Conda Environment Containing All Required Dependecies

`conda env create --name fcnkeras --file=environment.yml`

# Missing Traffic Signs Video Dataset Samples

### Missing Signs

![](https://github.com/iHubData-Mobility/public-CueCAn/blob/main/media/gifs/missing_samples.gif)

## Missing Traffic Signs Video Dataset (MTSVD) 

### Marked Missing Intervals

![](https://github.com/iHubData-Mobility/public-CueCAn/blob/main/media/gifs/missingmarked.gif)

### Traffic Sign Annotations and Tracks

![](https://github.com/iHubData-Mobility/public-CueCAn/blob/main/media/gifs/annotations.gif)

### CueCAn GradCam

![](https://github.com/iHubData-Mobility/public-CueCAn/blob/main/media/gifs/gradcam.gif)

### Video Result

![](https://github.com/iHubData-Mobility/public-CueCAn/blob/main/media/gifs/realtime.gif)

### Running GradCam for Segmentation Task

Due to memory constraints in google colab, the gradCam for segmentation task can be run by the following:

```
git clone https://github.com/iHubData-Mobility/public-CueCAn.git
cd public-CueCAn
conda env create -f environment.yml
conda activate fcnkeras
gdown https://drive.google.com/uc?id=1xr0v0f-tMVCE-s_OCz5YCnkTXJW8HsS3
python gradSeg.py --model_dir cuecan_segmenter.h5 --img_dir images/context
```
And the segmentation outputs, would be stored in directory `outSegGrad/`

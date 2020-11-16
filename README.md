# EGGAN: Fine-Grained Expression Manipulation via Structured Latent Space
This repository implements the training and testing of EGGAN for "[Fine-Grained Expression Manipulation via Structured Latent Space](https://ieeexplore.ieee.org/document/9102852)". It offers the original implementation of the paper in PyTorch. 

## Dependencies
```
pip install -r requirement.txt
```
## Dataset
The data of DISFA dataset can be download from [here](https://drive.google.com/drive/folders/1hapEvSWdHrKwbM8_3vt1kNEVvpd5bS_J?usp=sharing). Please unzip the data.zip under the "data/DISFA" folder, in which the path file of DISFA contains each frame of videos.

## Pretrain process
The pretrained model of the Identity Classifier can be download from [here](https://drive.google.com/drive/folders/16TU0Q5NJ3RsfVIlOl2i5dCckf9av-tOH?usp=sharing). You can train with other datasets by:
```
python3 pretrain.py
```
And modify the trainning details.
## Training
```
bash runner.sh
```
## Testing
After trainning, you can test the model by loading the specified model.
```
bash test.sh
```
## Citation
```
@inproceedings{tang2020fine,
  title={Fine-Grained Expression Manipulation Via Structured Latent Space},
  author={Tang, Junshu and Shao, Zhiwen and Ma, Lizhuang},
  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

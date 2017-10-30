# Cdiscount-caffe
Caffe codes for Kaggle Cdiscount Challenge.

## Usage
1. Download datasets from [Cdiscount’s Image Classification Challenge](https://www.kaggle.com/c/cdiscount-image-classification-challenge/data)
2. Check if data folders are correct
3. Download pre-trained models from [DenseNet 121](https://drive.google.com/open?id=0B7ubpZO7HnlCcHlfNmJkU2VPelE) [SE-ResNet-50](https://drive.google.com/open?id=0B7ubpZO7HnlCWkwtSG5CdXBKcmc)
4. Pre-process data: `python3 preprocessing.py`
5. Generate LMDB: `python3 convert2lmdb.py`
6. Run `sh train.sh` to start training

## Resources
* [SENet-Caffe](https://github.com/shicai/SENet-Caffe)
* [DenseNet-Caffe](https://github.com/shicai/DenseNet-Caffe)
* [DenseNet Space Efficient Implementation In Caffe](https://github.com/Tongcheng/DN_CaffeScript)
* [Xception-caffe](https://github.com/yihui-he/Xception-caffe)
* [Caffe-model](https://github.com/soeaver/caffe-model#caffe-model)

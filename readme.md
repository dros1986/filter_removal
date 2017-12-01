Pytorch implementation of the paper:

**Artistic Photo Filter Removal Using CNNs**  
Journal of Electronic Imaging, SPIE  
[F. Piccoli](http://www.ivl.disco.unimib.it/people/flavio-piccoli/ "Flavio Piccoli"), [C. Cusano](http://www.ivl.disco.unimib.it/people/claudio-cusano/ "Claudio Cusano"), [S. Bianco](http://www.ivl.disco.unimib.it/people/simone-bianco/ "Simone Bianco"), [R. Schettini](http://www.ivl.disco.unimib.it/people/raimondo-schettini/ "Raimondo Schettini")

Usage:

```bash
# clone this repository
git clone --recursive https://github.com/dros1986/filter_removal.git
# download the dataset
wget https://drive.google.com/a/campus.unimib.it/uc?export=download&confirm=XAOn&id=1vvLAO__opCjgLfRjAjW3WPWJHNiiVLbs
# unzip the file
unzip file.zip -d ./datasets/
# lunch training
python main.py -degin 3 degout 3
# lunch test
python main.py -degin 3 degout 3 --regen ./checkpoint.pth
```

Input images
![input](https://github.com/dros1986/filter_removal/blob/master/images/input.png)

Output images
![output](https://github.com/dros1986/filter_removal/blob/master/images/output.png)

<video width="1292" height="1292" controls preload autoplay loop>
<source src="https://github.com/dros1986/filter_removal/blob/master/images/output.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

<video width="320" height="240" controls>
  <source src="https://github.com/dros1986/filter_removal/blob/master/images/output.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

## Parameters
| Name | Description | Default |
| ---- | ----------- | ------- |
| degin | Degree of the polynomial onto which the color transform will be estimated | 3 |
| degout | Degree of the polynomial onto which the color transform will be applied | 3 |
| patchsize | patchsize*patchsize is the number of pixels involved in each color transform | 8 |
| nrow | Batch size will be nrow*nrow | 5 |
| indir | Folder containing filtered images | ./datasets/places-instagram/images/ |
| gtdir | Folder containing original images | ./datasets/places-instagram/images_orig/ |
| train_list | txt containing train set filenames | ./datasets/places-instagram/train-list.txt |
| validation_list | txt containing validation set filenames | ./datasets/places-instagram/smallvalidation-list.txt |
| test_list | txt containing test set filenames | ./datasets/places-instagram/test-list.txt |

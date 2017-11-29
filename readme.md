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

## Exp Table
| Exp       | patchSize | Net In | Poly In | Range | Interpolation | TanH | ReLU |
| --------- | --------- | ------ | ------- | ----- | ------------- | ---- | ---- |
| exp2      | 8       | R,G,B                           | <-- | [-1,1] | Bilinear | Y | Y |
| exp3      | 8       | R,G,B                           | <-- | [-1,1] | Bilinear | N | Y |
| exp4      | 8       | R2,G2,B2,R,G,B                  | <-- | [-1,1] | Bilinear | N | Y |
| exp5      | 8       | R2,G2,B2,R,G,B,K                | <-- | [-1,1] | Bilinear | N | Y |
| exp6->11  | 8-> 256 | (R+G+B)^2+(R+G+B)+K             | <-- | [-1,1] | Bilinear | N | Y |
| exp12->17 | 8-> 256 | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K   | <-- | [-1,1] | Bilinear | N | Y |
| exp18     | 256     | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K   | <-- | [-1,1] | Bilinear | N | N |
| exp19     | 8       | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K   | <-- | [0,1]  | Bilinear | N | N |
| exp20     | 8       | R+G+B | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K | [0,1]  | Bilinear | N | N |
| exp21->26 | 8-> 256 | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K   | <-- | [0,1]  | Bilinear | N | N |

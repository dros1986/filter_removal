Clone this repository in this way:

```git
git clone --recursive https://github.com/dros1986/filter_removal.git
```

## Experiments
### Exp1
Patch-based version, no interpolation

Input images
![input](https://github.com/dros1986/filter_removal/blob/master/images/input.png)

Output images
![output](https://github.com/dros1986/filter_removal/blob/master/images/output.png)

Comparison
![input](https://github.com/dros1986/filter_removal/blob/master/images/comparison.gif)

## Exp Table
| Exp       | patchSize | Input | Range | Interpolation | TanH |
| --------- | --------- | ----- | ----- | ------------- | ---- |
| exp2      | 8       | R,G,B                         | [-1,1] | Bilinear | Y |
| exp3      | 8       | R,G,B                         | [-1,1] | Bilinear | N |
| exp4      | 8       | R2,G2,B2,R,G,B                | [-1,1] | Bilinear | N |
| exp5      | 8       | R2,G2,B2,R,G,B,K              | [-1,1] | Bilinear | N |
| exp6->11  | 8-> 256 | (R+G+B)^2+(R+G+B)+K           | [-1,1] | Bilinear | N |
| exp12->17 | 8-> 256 | (R+G+B)^3+(R+G+B)^2+(R+G+B)+K | [-1,1] | Bilinear | N |

# GaussianUpsampling
The tensorflow 2 Implementations of Gaussian up-sampling

## Summary
Bilinear up-sampling is a special case of Gaussian up-sampling with kernel_size of 4 when scale factor is 2.
 
## Cautions
* scale factor cannot be too large, otherwise the norm weights will be calculated incorrectly.
* set the smooth parameter to False, which has a sharpening effect.
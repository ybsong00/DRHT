This is the implementation of the DRHT paper. The project page can be found here:

https://ybsong00.github.io/cvpr18_imgcorrect/index.html

Because of the size limit we put ldr2hdr model eslewhere at 
https://drive.google.com/open?id=138JfKA5QzjDu78PLf6Ih9t5Qh2bBMvhI. 

You need to download it at first and put it right under checkpoint folder.

The illustration of the files and folders.
############### folders ################
checkpoint      ---         pre-trained models
input           ---         input ldr images
hdr_output      ---         hdr files
samples         ---         ldr results
############### .py files ################
ldr2hdr.py and hdr2ldr.py define the ldr2hdr and hdr2ldr networks respectively, and 
ldr2hdr_test.py and hdr2ldr_test.py provide simple evaluation.

############### notes ################
1. The ldr2hdr part is based on the Siggraph Asia 17 paper "HDR image reconstruction from a single exposure using deep CNNs".
2. The hdr2ldr part performs better when using large batch_size. 

<p>If you find the code useful, please cite our paper:</p>

<pre><code>@inproceedings{yang-cvpr18-DRHT,
    author = {Yang, Xin and Xu, Ke and Song, Yibing and Zhang, Qiang and Wei, Xiaopeng and Rynson, Lau},
    title = {Image Correction via Deep Reciprocating HDR Transformation},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2018},
  }
</code></pre>

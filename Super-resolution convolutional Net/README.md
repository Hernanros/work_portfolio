# Super Resolution Netwoks
this project was created as a part of an Homework assignment for Y-Data Deep learning course submitted by Shaul Solomon and myself. <br>
The purpose of this assignment is to get familiar with construction and training of fully convolutional networks. we will specifically use the task of image super-resolution, and we’ll construct several different architectures and compare the results achieved by each of them. <br>
in this projects we expirimented with several different types of architechtures for Fully convonutional networks used for image quality improvement assignment.<br>
in this project we built 6 different models and compared their preformence on taking 72 * 72 images and producing a 144 * 144 and 288 * 288 images.
All models are available in the models [folder](https://github.com/Hernanros/work_portfolio/tree/master/Super-resolution%20convolutional%20Net/models):<br>
### Model 1 - Benchmark<br>
fully convolutional model with sampling<br>
### Model 2 - Fully conv. model with 2 different outputs
fully convolutional model that output upsampled images both in 144*144 and 288*288 resolution
### Model 3 - Residual blocks<br>
fully convolutional model with residual blocks that output upsampled images both in 144*144 and 288*288 resolution
### Model 4 - a dilated (Atrous) convolutions <br>
fully convolutional model with dilated blocks that output upsampled images both in 144*144 and 288*288 resolution
### Model 5 - tranfer learning
model that concatenates convolutions with pretrained weights from [VGG 16](https://neurohive.io/en/popular-networks/vgg16/)
### Model 6 - Pixel shuffle
model that uses pixel shuffle techniques as generaliztion for upsampling

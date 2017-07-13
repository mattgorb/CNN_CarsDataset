

Car classification.  Fine-tuned InceptionV3 convolutional neural network based on this dataset:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html

Does pretty well, >97% on validation and test sets.   

Tech: 

AWS p2.xlarge ubuntu linux instance (GPU support)
cuda/cudnn
Python/Numpy
Python virtual environment.  
Keras
TensorFlow
OpenCV

Setting up a computing instance on Amazon was a pain in the ass.  I'll highlight the steps here.  

1.  Request P2.xlarge instance on aws.  You have to write a message to customer support and they'll usually get you approved within a business day.  

2.  Create a new Ubuntu instance on ec2 console.  

3.  You'll probably want to extend your partition, especially if you want to hold training data and various ML tech on here.  Link for that-
http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/storage_expand_partition.html

4.  Download CUDA and CUDNN for nvidia GPU support.  
CUDA install command:
sudo apt install nvidia-cuda-toolkit
Link for CUDNN:  https://developer.nvidia.com/rdp/cudnn-download

5.  Download python on ec2 instance and figure out how to setup a python virtual environment.  Google it.

6.  Download keras. 

7.  Download required python libraries such as numpy.  
  
8.  OpenCV: sudo apt-get install python-opencv.  You might 

9.  Download tensor flow.  I'm using pip.  Then configure keras to use the tensorflow backend.  I tried for a half of a day to get Theano backend working with GPU support and failed miserably.  TF pip link: 
https://www.tensorflow.org/versions/r0.11/get_started/os_setup#pip_installation


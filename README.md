### setup :
unzip mnist_png.zip and place it in the same folder it was originally in.
The folder structure should be DCGAN/data/mnist_png/mnist_png/

#### To use the trained model :
Head on over to https://drive.google.com/drive/folders/1R-PC2zI3KVMCxVkxe4vMQAoPe9TwULZI?usp=sharing and 
download generator_wgan_no_batchnorm_no_conv.ser and critic_wgan_no_conv.ser which are the currently working models,
and place it in the /models directory. 

The models were compiled on oracle open-jdk-21, so if it isn't working for you, switch to this jdk.

The working implementation is in the file called "WGAN_No_Conv.java"

### To run the trained model and see its outputs :
run Generator_Usage.java . This will load the trained model and generate 8s;

### note :
Generator_Implementation_Without_Batchnorm.java is the current generator network being used. 
The one with Batch Normalization is having a vanishing gradient problem that has to be fixed.

Generator_Implementation_Without_Batchnorm.java has a main function which shows an example of training
the network separately(without a discriminator) to output a single handdrawn digit by simply using RMSE against the single target image.

DCGAN_Implementation.java is the original DCGAN implementation that was used to train the model. 
But it doesn't work because of over sensitivity to hyperparameters, and just the fact that it is a bad architecture to begin with.

### To calculate the FID score : 
First, install the pytorch-fid package by running:
```bash
pip install pytorch-fid --user
```

Then,
from the project directory, run:
```bash
python -m pytorch_fid outputs/generator_outputs training_dataset
```
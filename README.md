Generator_Implementation_Without_Batchnorm.java is the current generator network being used. The one with Batch Normalization is having a vanishing gradient problem that has to be fixed.

Generator_Implementation_Without_Batchnorm.java has a main function which shows an example of training the network separately(without a discriminator) to output a single handdrawn digit by simply using RMSE against the single target image.

DCGAN_Implementation.java has a simple training loop where the discriminator is no longer trained if its accuracy is too high.

However, the problem now is that the generator does not train properly even though the discriminator is able to reach 100% accuracy. The generator is simply never able to fool the discriminator.
There is likely something wrong with the way the discriminator is feeding its gradients to the generator during back propagation or there is something wrong with the loss function used.
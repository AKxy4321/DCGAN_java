package DCGAN;

import DCGAN.networks.Generator_Implementation_Without_Batchnorm;
import DCGAN.util.MiscUtils;

import static DCGAN.util.MiscUtils.getBufferedImage;
import static DCGAN.util.SerializationUtils.loadObject;

public class GeneratorUsage {
    /**
     * This class includes code to load a trained generator model and generate images using it.
     * */


    public static void main(String[] args) {
        Generator_Implementation_Without_Batchnorm generator =
                (Generator_Implementation_Without_Batchnorm) loadObject("models/generator_wgan_no_batchnorm_no_conv.ser");
        MiscUtils.saveImage(getBufferedImage(generator.generateImage()), "outputs/fake_eight1.png");
        MiscUtils.saveImage(getBufferedImage(generator.generateImage()), "outputs/fake_eight2.png");
    }
}

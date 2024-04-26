package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DCGAN_Implementation {

    public static void main(String[] args) {
        DCGAN_Implementation dcgan = new DCGAN_Implementation();
        dcgan.dcgan_execute();
    }


    public double calculateAccuracy(double[] real_outputs, double[] fake_outputs) {
        double accuracy = 0;
        for (int j = 0; j < real_outputs.length; j++) {
            if (real_outputs[j] > 0.5) {
                accuracy += 1;
            }
            if (fake_outputs[j] < 0.5) {
                accuracy += 1;
            }
        }
        return accuracy;
    }


    public void dcgan_execute() {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
        int train_size = 3200;
        int label = 3;
        double learning_rate_gen = 0.005;
        double learning_rate_disc = 0.005;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation generator = new Generator_Implementation();
        System.out.println("Loading Images");

        int batch_size = 8;
        // minibatch gradient descent
        for (int epochs = 0; epochs < 50; epochs++) {
            int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            logger.log(Level.INFO, "Epoch " + epochs);
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {
                double[][][][] fakeImages = new double[batch_size][1][28][28];
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++) {
                    BufferedImage img = DCGAN.UTIL.mnist_load_index(label, index[label]);
                    realImages[real_idx] = new double[][][]{UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(img))};
                    index[label] += 1;
                }
                for (int j = 0; j < batch_size; j++) {
                    fakeImages[j] = generator.generateImage();
                    System.out.print(j + " ");
                }
                System.out.println("Finished loading images");
                double[][] discriminator_fake_outputs = new double[batch_size][1];
                double[][] discriminator_real_outputs = new double[batch_size][1];
                double[] gen_losses = new double[batch_size];
                double[] disc_losses = new double[batch_size];


                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    System.out.println(realImages[img_idx][0].length + " " + realImages[img_idx][0][0].length);
                    double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx][0]);

                    double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                    discriminator_real_outputs[img_idx] = discriminator_output_real;
                    discriminator_fake_outputs[img_idx] = discriminator_output_fake;

                    double gen_loss = UTIL.gen_loss(discriminator_output_fake);
                    double disc_loss = UTIL.disc_loss(discriminator_output_real, discriminator_output_fake);
                    gen_losses[img_idx] = gen_loss;
                    disc_losses[img_idx] = disc_loss;

                    System.out.print(img_idx + " ");
                }

                logger.log(Level.INFO, "Epoch:" + epochs + " batch:" + batch_idx);
                logger.log(Level.INFO, "Gen_Loss " + DCGAN.UTIL.mean(gen_losses));
                logger.log(Level.INFO, "Disc_Loss " + DCGAN.UTIL.mean(disc_losses));

                double[][] disc_output_gradients = new double[batch_size][1];
                double[][][][] gen_output_gradients = new double[batch_size][1][28][28];

                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    double[] discriminator_output_fake = discriminator_fake_outputs[img_idx];
                    double[] discriminator_output_real = discriminator_real_outputs[img_idx];

//                System.out.println("computing discriminator gradients");
                    // gradient wrt both real and fake output
                    double[] discriminator_outputs = new double[]{discriminator_output_real[0], discriminator_output_fake[0]};
                    double[] dg = new double[]{UTIL.gradientDiscriminatorMSE(discriminator_outputs, new double[]{1.0, 0.0})};
                    disc_output_gradients[img_idx] = dg; //UTIL.negate(dg);

                    //gradient wrt only fake output
                    double[] disc_gradient_fake = UTIL.negate(dg); // new double[]{UTIL.gradientDiscriminatorMSE(discriminator_output_fake, new double[]{0.0})});
                    disc_gradient_fake[0] *= 5;
                    // do one forward pass so that convolution layer stores this image
                    discriminator.getOutput(fakeImages[img_idx][0]);

                    double[][][] fake_back_gradient = discriminator.backward(disc_gradient_fake);
                    gen_output_gradients[img_idx] = fake_back_gradient;
                }

                // back prop with update parameters
                double[] disc_output_gradient = UTIL.mean_1st_layer(disc_output_gradients);
                double[][][] gen_output_gradient = UTIL.mean_1st_layer(gen_output_gradients);

                // update discriminator
                discriminator.updateParameters(disc_output_gradient, learning_rate_disc);


                // update generator
                generator.updateParameters(gen_output_gradient, learning_rate_gen);

                // generate image
                BufferedImage image = DCGAN.UTIL.getBufferedImage(generator.generateImage());

                DCGAN.UTIL.saveImage(image, "generated_image.png");
            }
        }
    }

    public static double lossDiscriminator(double[] real_output, double[] fake_output) {
        double loss = 0;
        for (int i = 0; i < real_output.length; i++) {
            loss += -Math.log(real_output[i]) - Math.log(1 - fake_output[i]);
        }
        return loss / real_output.length;
    }

    public static double[] computeGradientDiscriminator(double[] real_output, double[] fake_output) {
        double[] gradient = new double[real_output.length];
        double epsilon = 1e-4F;
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = -(1.0 / (real_output[i] + epsilon) + (1 / (fake_output[i] + epsilon)));
        }
        return gradient;
    }

    public static double[] computeGradientDiscriminatorWRTFake(double[] fake_output) {
        double[] gradient = new double[fake_output.length];
        double epsilon = 1e-1F;
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] += (1 / (fake_output[i] - 1 + epsilon));
        }
        return gradient;
    }
}
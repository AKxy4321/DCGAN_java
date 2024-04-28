package DCGAN;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import static DCGAN.UTIL.gradientBinaryCrossEntropy;
import static DCGAN.UTIL.lossBinaryCrossEntropy;

public class DCGAN2 {

    public static void main(String[] args) {
        DCGAN2 dcgan = new DCGAN2();
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
        double learning_rate_gen = -1e-4;
        double learning_rate_disc = 1e-4;
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

                double accuracy = 0.0;

                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    System.out.println(realImages[img_idx][0].length + " " + realImages[img_idx][0][0].length);
                    double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx][0]);
                    double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                    accuracy += calculateAccuracy(discriminator_output_real, discriminator_output_fake);

                    discriminator_real_outputs[img_idx] = discriminator_output_real;
                    discriminator_fake_outputs[img_idx] = discriminator_output_fake;

                    double gen_loss = generatorLoss(discriminator_output_fake);
                    double disc_loss = discriminatorLoss(discriminator_output_real, discriminator_output_fake);
                    gen_losses[img_idx] = gen_loss;
                    disc_losses[img_idx] = disc_loss;

                    System.out.print(img_idx + " ");
                }

                logger.log(Level.INFO, "Epoch:" + epochs + " batch:" + batch_idx);
                logger.log(Level.INFO, "Gen_Loss " + DCGAN.UTIL.mean(gen_losses));
                logger.log(Level.INFO, "Disc_Loss " + DCGAN.UTIL.mean(disc_losses));
                logger.log(Level.INFO, "Accuracy " + accuracy / (batch_size * 2));

                double[][] disc_output_gradients = new double[batch_size][1];
                double[][][][] gen_output_gradients = new double[batch_size][1][28][28];

                for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                    double[] discriminator_output_fake = discriminator_fake_outputs[img_idx];
                    double[] discriminator_output_real = discriminator_real_outputs[img_idx];

//                System.out.println("computing discriminator gradients");
                    // gradient wrt both real and fake output
                    double[] output_gradient = gradientBinaryCrossEntropy(
                            batch_idx % 2 == 0 ? discriminator_output_real : discriminator_output_fake,
                            new double[]{batch_idx % 2 == 0 ? 1 : 0});
                    double[] dg = output_gradient; // discriminatorLossGradient(discriminator_output_real, discriminator_output_fake);
                    disc_output_gradients[img_idx] = dg; //UTIL.negate(dg);

                    //gradient wrt only fake output
                    double[] disc_gradient_fake = generatorLossGradientNew(discriminator_output_fake);//gradientBinaryCrossEntropy(discriminator_output_fake, new double[]{0});
                    disc_gradient_fake[0] *= 1;
                    // do one forward pass so that convolution layer stores this image
                    discriminator.getOutput(fakeImages[img_idx][0]);

                    double[][][] fake_back_gradient = discriminator.backward(disc_gradient_fake);
                    gen_output_gradients[img_idx] = fake_back_gradient;

                    if(img_idx == batch_size-1){
                        System.out.println("Discriminator output real: " + Arrays.toString(discriminator_output_real));
                        System.out.println("Discriminator output fake: " + Arrays.toString(discriminator_output_fake));
                        System.out.println("Discriminator gradient: " + Arrays.toString(dg));
                        System.out.println("Generator gradient: " + Arrays.toString(disc_gradient_fake));
                    }
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
                System.out.println("DCGAN2 saving image");
                DCGAN.UTIL.saveImage(image, "output2.png");
                System.out.println("Saved image");
            }
        }
    }


    public double generatorLoss(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, 1);
        return lossBinaryCrossEntropy(fake_output, ones);
    }

    public double discriminatorLoss(double[] real_output, double[] fake_output) {
        double[] ones = new double[real_output.length];
        double[] zeros = new double[real_output.length];
        Arrays.fill(ones, 1);

        double real_loss = lossBinaryCrossEntropy(real_output, ones);
        double fake_loss = lossBinaryCrossEntropy(fake_output, zeros);

        return real_loss + fake_loss;
    }

    public double[] discriminatorLossGradient(double[] real_output, double[] fake_output) {
        double[] ones = new double[real_output.length];
        double[] zeros = new double[real_output.length];
        Arrays.fill(ones, 1);
        Arrays.fill(zeros, 0);
        double[] real_gradient = gradientBinaryCrossEntropy(real_output, ones);
        double[] fake_gradient = gradientBinaryCrossEntropy(fake_output, zeros);
        double[] gradient = new double[real_output.length];
        for (int i = 0; i < real_output.length; i++) {
            gradient[i] = real_gradient[i] + fake_gradient[i];
        }
        return gradient;
    }

    public double generatorLossNew(double[] fake_output) {
        // loss function : -log(D(G(z))
        return -Math.log(fake_output[0] + epsilon);
    }
    public double[] generatorLossGradientNew(double[] fake_output){
        // loss function : -log(D(G(z)))
        double[] gradient = new double[fake_output.length];
        for(int i = 0;i<fake_output.length;i++){
            gradient[i] = 1/(fake_output[i] + epsilon);
        }
        return gradient;
    }

    public double[] generatorLossGradient(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, 1);
        return gradientBinaryCrossEntropy(fake_output, ones);
    }

    public double epsilon = 0.00001;

}
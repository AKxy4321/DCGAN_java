package DCGAN.networks;

import DCGAN.UTIL;
import DCGAN.XavierInitializer;

import java.awt.image.BufferedImage;
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


    /*
    * tanh_in_gradient: 0.004388311190731719
leakyrelu3_in_gradient: 0.795224583610218
tconv2_in_gradient: -3.9007676460698053
leakyrelu2_in_gradient: 0.6127335342649229
tconv1_in_gradient: 10.430742366071163
leakyrelu_in_gradient: 9.043740669077065
leakyrelu_in_gradient_flattened: 9.043740669077065
dense_in_gradient: 1.1095968969529342
Generator loss function gradient: [3.2944894818692116]
    * */

    public void dcgan_execute() {
        Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());
        int train_size = 1;
        int label = 10;
        double learning_rate_gen = 1 * 1e-1;
        double learning_rate_disc = 1 * 1e-3;
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm();
        generator.verbose = true;
        discriminator.verbose = false;
        System.out.println("Loading Images");

        int batch_size = 1; // batch size of 1 for sgd

        boolean disc_frozen = false;

        // training it on only one image
        BufferedImage real_img = DCGAN.UTIL.mnist_load_index(label, 0);
        UTIL.saveImage(real_img, "real_image.png");

        // minibatch gradient descent
        for (int epochs = 0; epochs < 1000000; epochs++) {
            int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            logger.log(Level.INFO, "Epoch " + epochs);
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {

                // Load images
                double[][][][] fakeImages = new double[batch_size][1][28][28];
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++) {
//                    BufferedImage real_img = DCGAN.UTIL.mnist_load_index(label, index[label]);
//                    UTIL.saveImage(real_img, "real_image.png");
                    realImages[real_idx] = new double[][][]{UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(real_img))};
                    index[label] += 1;
                }
                for (int j = 0; j < batch_size; j++) {
                    fakeImages[j] = generator.generateImage();
                    System.out.print(j + " ");
                }
                System.out.println("Finished loading images");


                double accuracy = 0.0;


                double disc_loss = 0.0;
                double gen_loss = 0.0;
                // calculate losses
                calc_losses:
                {
                    double[] gen_losses = new double[batch_size];
                    double[] disc_losses = new double[batch_size];
                    for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                        System.out.println(realImages[img_idx][0].length + " " + realImages[img_idx][0][0].length);
                        double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx][0]);
                        double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                        accuracy += calculateAccuracy(discriminator_output_real, discriminator_output_fake);

                        gen_losses[img_idx] = generatorLossNew(discriminator_output_fake);
                        disc_losses[img_idx] = discriminatorLoss(discriminator_output_real, discriminator_output_fake);

                        System.out.print(img_idx + " ");
                    }

                    disc_loss = UTIL.mean(disc_losses);
                    gen_loss = UTIL.mean(gen_losses);

                    logger.log(Level.INFO, "Epoch:" + epochs + " batch:" + batch_idx);
                    logger.log(Level.INFO, "Gen_Loss " + gen_loss);
                    logger.log(Level.INFO, "Disc_Loss " + disc_loss);
                    logger.log(Level.INFO, "Accuracy " + accuracy / (batch_size * 2));
                }

                double[][] disc_output_gradients = new double[batch_size][1];

                double[] expected_real_output = {1.0};
                double[] expected_fake_output = {0.0};

//                double min_disc_loss = 0.1, escape_disc_loss = 0.3;
//                if (disc_frozen) {
//                    if (disc_loss > escape_disc_loss)
//                        disc_frozen = false;
//                }else{
//                    if (disc_loss < min_disc_loss)
//                        disc_frozen = true;
//                }
                logger.info("disc_frozen : "+ disc_frozen);

                if (!disc_frozen) {
                    // train on real images
                    train_disc_on_real:
                    {
                        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                            double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx][0]);

                            double[] output_gradient = gradientBinaryCrossEntropy(discriminator_output_real, expected_real_output);
                            disc_output_gradients[img_idx] = output_gradient;

                            if (img_idx == batch_size - 1) {
                                System.out.println("Discriminator output real: " + Arrays.toString(discriminator_output_real));
                            }
                        }

                        double[] disc_output_gradient = UTIL.mean_1st_layer(disc_output_gradients);
                        discriminator.updateParameters(disc_output_gradient, learning_rate_disc);
                    }

                    // train with fake images
                    train_disc_on_fake:
                    {
                        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                            double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                            double[] output_gradient = gradientBinaryCrossEntropy(discriminator_output_fake, expected_fake_output);
                            disc_output_gradients[img_idx] = output_gradient;

                            if (img_idx == batch_size - 1) {
                                System.out.println("Discriminator output fake: " + Arrays.toString(discriminator_output_fake));
                            }
                        }

                        double[] disc_output_gradient = UTIL.mean_1st_layer(disc_output_gradients);
                        discriminator.updateParameters(disc_output_gradient, learning_rate_disc);
                    }
                }

                // train generator
                train_generator:
                {
                    for (int img_idx = 0; img_idx < batch_size; img_idx++) {
                        double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx][0]);

                        double[] output_gradient = generatorLossGradientNew(discriminator_output_fake);

                        disc_output_gradients[img_idx] = output_gradient;

                        if (img_idx == batch_size - 1) {
                            System.out.println("Discriminator output fake: " + Arrays.toString(discriminator_output_fake));
                        }
                    }

                    double[] disc_output_gradient = UTIL.mean_1st_layer(disc_output_gradients);
                    double[][][] generatorOutputGradient = discriminator.backward(disc_output_gradient);
                    generator.updateParameters(generatorOutputGradient, learning_rate_gen);

                    System.out.println("Generator loss function gradient: " + Arrays.toString(disc_output_gradient));
                }

                // generate image
                BufferedImage image = DCGAN.UTIL.getBufferedImage(generator.generateImage());
                DCGAN.UTIL.saveImage(image, "generated_image.png");
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
        return Math.log(fake_output[0] + epsilon);
    }

    public double[] generatorLossGradientNew(double[] fake_output) {
        // loss function : -log(D(G(z)))
        double[] gradient = new double[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] = -1 / (fake_output[i] + epsilon);
        }

        // changed the loss gradient
        return gradient;
    }

    public double[] generatorLossGradient(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, 1);
        return gradientBinaryCrossEntropy(fake_output, ones);
    }

    public double lossBinaryCrossEntropy(double[] outputs, double[] labels) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += labels[i] * Math.log(outputs[i] + epsilon) + (1 - labels[i]) * Math.log(1 - outputs[i] + epsilon);
        }
        return -loss / outputs.length;
    }

    public double[] gradientBinaryCrossEntropy(double[] outputs, double[] labels) {
        double[] gradient = new double[outputs.length];
        for (int i = 0; i < outputs.length; i++) {
            gradient[i] = (outputs[i] - labels[i]) / (outputs[i] * (1 - outputs[i]) + epsilon);
        }
        return gradient;
    }

    public double epsilon = 0.00000001;

}
package DCGAN;

import DCGAN.networks.Discriminator_Implementation;
import DCGAN.networks.Generator_Implementation;
import DCGAN.util.MiscUtils;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.logging.Logger;

import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.TrainingUtils.gradientBinaryCrossEntropy;
import static DCGAN.util.TrainingUtils.lossBinaryCrossEntropy;


public class DCGAN_Implementation {
    final private static Logger logger = Logger.getLogger(DCGAN_Implementation.class.getName());

    int train_size = 100;
    int label = 9;
    double learning_rate_gen = 1 * 1e-1;
    double learning_rate_disc = 1 * 1e-3;
    int batch_size = 8;
    int minimum_gen_training_start_epoch = 100;

    private double disc_loss, gen_loss, accuracy;

    Discriminator_Implementation discriminator = new Discriminator_Implementation();
    Generator_Implementation generator = new Generator_Implementation(batch_size);

    double[] expected_real_output = {1.0};
    double[] expected_fake_output = {0.0};

    public static void main(String[] args) {
        DCGAN_Implementation dcgan = new DCGAN_Implementation();
        dcgan.dcgan_execute();
    }

    public void dcgan_execute() {
        boolean disc_frozen = false;

        generator.verbose = true;
        discriminator.verbose = false;

        // minibatch gradient descent
        for (int epochs = 0; epochs < 1000000; epochs++) {
            int img_index = 0;
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {

                // Load images
                double[][][][] fakeImages = generator.forwardBatch();
                double[][][][] realImages = new double[batch_size][1][28][28];

                for (int real_idx = 0; real_idx < batch_size; real_idx++) {
                    BufferedImage real_img = MiscUtils.mnist_load_index(label, img_index);
                    realImages[real_idx] = new double[][][]{zeroToOneToMinusOneToOne(img_to_mat(real_img))};

                    if (real_idx == 0)
                        saveImage(real_img, "real_image.png");

                    img_index++;
                }

                System.out.println("Finished loading images");

                // calculating and displaying our model metrics
                calculateModelMetrics(realImages, fakeImages);

                logger.info("Epoch : " + epochs + " batch:" + batch_idx);
                logger.info("Gen_Loss : " + gen_loss);
                logger.info("Disc_Loss : " + disc_loss);
                logger.info("Accuracy : " + accuracy);


                //deciding whether to freeze discriminator weights or not
                double min_disc_loss = 0.05, escape_disc_loss = 0.1;
                if (epochs > minimum_gen_training_start_epoch) {
                    if (disc_frozen) {
                        if (disc_loss > escape_disc_loss)
                            disc_frozen = false;
                    } else {
                        if (disc_loss < min_disc_loss)
                            disc_frozen = true;
                    }
                }
                logger.info("disc_frozen : " + disc_frozen);


                // train discriminator
                if (!disc_frozen)
                    train_discriminator(realImages, fakeImages, learning_rate_disc);

                // train generator
                if (epochs > minimum_gen_training_start_epoch)
                    train_generator(fakeImages, learning_rate_gen);
                else
                    logger.info("gen_frozen : " + true);

                // generate image
                saveImage(getBufferedImage(generator.generateImage()), "generated_image_dcgan.png");
            }
        }
    }

    private void calculateModelMetrics(double[][][][] realImages, double[][][][] fakeImages) {
        double[] gen_losses = new double[batch_size];
        double[] disc_losses = new double[batch_size];
        double num_correct_predictions = 0.0;
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx]);
            double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx]);

            num_correct_predictions += (discriminator_output_fake[0] <= 0.5 ? 1 : 0) + (discriminator_output_real[0] > 0.5 ? 1 : 0);

            gen_losses[img_idx] = generatorLossNew(discriminator_output_fake);
            disc_losses[img_idx] = discriminatorLoss(discriminator_output_real, discriminator_output_fake);

            System.out.print(img_idx + " ");
        }
        accuracy = num_correct_predictions / (batch_size * 2.0);

        disc_loss = MiscUtils.mean(disc_losses);
        gen_loss = MiscUtils.mean(gen_losses);
    }


    public void train_discriminator(double[][][][] realImages, double[][][][] fakeImages, double learning_rate) {
        // train on real images
        train_discriminator(realImages, expected_real_output, learning_rate);

        // train with fake images
        train_discriminator(fakeImages, expected_fake_output, learning_rate);
    }

    public void train_discriminator(double[][][][] images, double[] expected_output, double learning_rate) {
        double[][] disc_output_gradients = new double[batch_size][];

        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            double[] discriminator_output = discriminator.getOutput(images[img_idx]);
            disc_output_gradients[img_idx] = gradientBinaryCrossEntropy(discriminator_output, expected_output);

            if (img_idx == batch_size - 1)
                System.out.println("Discriminator output : " + Arrays.toString(discriminator_output) + " expected : " + expected_output[0]);
        }

        discriminator.updateParameters(MiscUtils.mean_1st_layer(disc_output_gradients), learning_rate);
    }

    public void train_generator(double[][][][] fakeImages, double learning_rate) {
        double[][][][] generatorOutputGradients = new double[batch_size][][][];
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx]);
            double[] disc_output_gradient_fake = generatorLossGradientNew(discriminator_output_fake);

            generatorOutputGradients[img_idx] = discriminator.backward(disc_output_gradient_fake);
        }

        generator.updateParametersBatch(generatorOutputGradients, learning_rate);
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
        // loss function : log(D(G(z)) which we want to maximize
        return Math.log(fake_output[0] + epsilon);
    }

    public double[] generatorLossGradientNew(double[] fake_output) {
        // loss function : -log(D(G(z)))
        double[] gradient = new double[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] = -1.0 / (fake_output[i] + epsilon);
        }

        // we want to ascend the gradient of the loss function, so
        // the loss gradient is multiplied by -1 inorder to do gradient descent in the opposite direction
        return gradient;
    }

    public double[] generatorLossGradient(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, 1);
        return gradientBinaryCrossEntropy(fake_output, ones);
    }


    private static final double epsilon = 0.00000001;

}
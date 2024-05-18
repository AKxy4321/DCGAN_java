package DCGAN;

import DCGAN.networks.Discriminator_Implementation;
import DCGAN.networks.Generator_Implementation;
import DCGAN.networks.Generator_Implementation_Without_Batchnorm;
import DCGAN.util.MiscUtils;
import DCGAN.util.TrainingUtils;

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
    double learning_rate_gen = 1 * 1e-3;
    double learning_rate_disc = 1 * 1e-3;
    int batch_size = 8;

    private double disc_loss, gen_loss, accuracy;

    Discriminator_Implementation discriminator = new Discriminator_Implementation(batch_size, learning_rate_disc);
    Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(batch_size, learning_rate_gen);

    // label smoothing for regularization so that the discriminator doesn't become too confident
    double[] expected_real_output = {0.98};
    double[] expected_fake_output = {0.05};

    public static void main(String[] args) {
        DCGAN_Implementation dcgan = new DCGAN_Implementation();
        dcgan.dcgan_execute();
    }

    public void dcgan_execute() {
        generator.verbose = true;
        discriminator.verbose = false;

        // we want to train the dcgan system to replicate this single image for now
        BufferedImage real_img = MiscUtils.mnist_load_index(label, 1);

        MiscUtils.saveImage(real_img, "real_image.png");

        double[][][] real_img_array = new double[][][]{zeroToOneToMinusOneToOne(img_to_mat(real_img))};

        // minibatch gradient descent
        for (int epochs = 0; epochs < 1000000; epochs++) {
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {

                // Load images
                double[][][][] fakeImages = generator.forwardBatch();
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++)
                    realImages[real_idx] = real_img_array;


                // calculating and displaying our model metrics
                calculateModelMetrics(realImages, fakeImages);

                logger.info("Epoch : " + epochs + " batch:" + batch_idx);
                logger.info("Gen_Loss : " + gen_loss);
                logger.info("Disc_Loss : " + disc_loss);
                logger.info("Accuracy : " + accuracy);

                // train discriminator
                if (accuracy < 0.99) // we don't want to train the discriminator too much
                    train_discriminator(realImages, fakeImages);

                // train generator
                if (epochs > 0) // we start training the generator only after the discriminator has learned something at the start
                    train_generator(fakeImages);

                // generate image
                double[][][] gen_img = generator.generateImage();
                saveImage(getBufferedImage(gen_img), "generated_image_dcgan.png");
                logger.info("RMSE with desired image : " + TrainingUtils.lossRMSE(gen_img, real_img_array));
            }
        }
    }


    public void train_discriminator(double[][][][] realImages, double[][][][] fakeImages) {
        // train on real images
        train_discriminator(realImages, expected_real_output);

        // train with fake images
        train_discriminator(fakeImages, expected_fake_output);
    }

    public void train_discriminator(double[][][][] images, double[] expected_output) {
        double[][] disc_outputs = discriminator.forwardBatch(images);

        double[][] disc_output_gradients = new double[batch_size][];
        for (int img_idx = 0; img_idx < batch_size; img_idx++)
            disc_output_gradients[img_idx] = gradientBinaryCrossEntropy(disc_outputs[img_idx], expected_output);

        discriminator.updateParametersBatch(disc_output_gradients);

        // for debugging
        System.out.println("Discriminator output : " + Arrays.toString(disc_outputs[0]) + " expected : " + expected_output[0]);
    }

    public void train_generator(double[][][][] fakeImages) {
        double[][] disc_fake_outputs = discriminator.forwardBatch(fakeImages);

        double[][] disc_fake_output_gradients = new double[batch_size][];
        for (int img_idx = 0; img_idx < batch_size; img_idx++)
            disc_fake_output_gradients[img_idx] = generatorLossGradient(disc_fake_outputs[img_idx]);

        double[][][][] disc_input_gradients_gen_output_gradients = discriminator.backwardBatch(disc_fake_output_gradients);

        // input to the discriminator is the output of the generator
        generator.updateParametersBatch(disc_input_gradients_gen_output_gradients);
    }

    private void calculateModelMetrics(double[][][][] realImages, double[][][][] fakeImages) {
        double[] gen_losses = new double[batch_size];
        double[] disc_losses = new double[batch_size];
        double num_correct_predictions = 0.0;
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            double[] discriminator_output_real = discriminator.getOutput(realImages[img_idx]);
            double[] discriminator_output_fake = discriminator.getOutput(fakeImages[img_idx]);

            num_correct_predictions += (discriminator_output_fake[0] <= 0.5 ? 1 : 0) + (discriminator_output_real[0] > 0.5 ? 1 : 0);

            gen_losses[img_idx] = generatorLoss(discriminator_output_fake);
            disc_losses[img_idx] = discriminatorLoss(discriminator_output_real, discriminator_output_fake);

            System.out.print(img_idx + " ");
        }
        accuracy = num_correct_predictions / (batch_size * 2.0);

        disc_loss = MiscUtils.mean(disc_losses);
        gen_loss = MiscUtils.mean(gen_losses);
    }

    public double generatorLoss(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, 1);
        return lossBinaryCrossEntropy(fake_output, ones);
    }

    public double[] generatorLossGradient(double[] fake_output) {
        double[] ones = new double[fake_output.length];
        Arrays.fill(ones, expected_real_output[0]);
        // we want to ascend the original gradient, so we just multiply the original gradient by -1 to descend it
        return MiscUtils.multiplyScalar(gradientBinaryCrossEntropy(fake_output, ones), -1);
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
        return -Math.log(fake_output[0] + epsilon);
    }

    public double[] generatorLossGradientNew(double[] fake_output) {
        // loss function : -log(D(G(z)))
        double[] gradient = new double[fake_output.length];
        for (int i = 0; i < fake_output.length; i++) {
            gradient[i] = -1.0 / (fake_output[i] + epsilon);
        }

        return gradient;
    }


    private static final double epsilon = 1e-6;

}
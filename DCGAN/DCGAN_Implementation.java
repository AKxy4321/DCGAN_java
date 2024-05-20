package DCGAN;

import DCGAN.networks.Discriminator_Implementation;
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

    int train_size = 1;
    int test_size = 100;
    int label = 9;
    double learning_rate_gen = 3 * 1e-3;
    double learning_rate_disc = 1 * 1e-4;
    int batch_size = 1;// switching to sgd

    private double disc_loss, gen_loss, accuracy;

    Discriminator_Implementation discriminator = new Discriminator_Implementation(batch_size, learning_rate_disc);
    Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(batch_size, learning_rate_gen);

    // label smoothing for regularization so that the discriminator doesn't become too confident
    double[] expected_real_output_disc = {0.9};
    double[] expected_fake_output_disc = {0.1};

    public static void main(String[] args) {
        DCGAN_Implementation dcgan = new DCGAN_Implementation();
        dcgan.dcgan_execute();
    }

    public void dcgan_execute() {
        generator.verbose = true;
        discriminator.verbose = false;

        // minibatch gradient descent
        for (int epochs = 0; epochs < 1000000; epochs++) {
            int index = 0;// reset our index to 0
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {
                // Load images
                double[][][][] fakeImages = generator.forwardBatch();
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++)
                    realImages[real_idx] = new double[][][]{addNoise(zeroToOneToMinusOneToOne(img_to_mat(mnist_load_index(label, index++))), 0.5)};

                // train discriminator
//                if (accuracy < 0.90) // we don't want to train the discriminator too much
                if (epochs % 2 == 0) // we train the discriminator every 2 epochs
                    train_discriminator(realImages, fakeImages);

                // train generator
//                if (accuracy > 0.5) // we start training the generator only after the discriminator has learned something at the start
                train_generator(fakeImages);

                // generate image
                double[][][] gen_img = generator.generateImage();
                saveImage(getBufferedImage(gen_img), "generated_image_dcgan.png");

                // calculating and displaying our model metrics
                calculateModelMetrics();
                logger.info("Epoch : " + epochs + " batch:" + batch_idx);
                logger.info("Gen_Loss : " + gen_loss);
                logger.info("Disc_Loss : " + disc_loss);
                logger.info("Accuracy : " + accuracy);

                if (epochs == 0) // for debugging
                    MiscUtils.saveImage(getBufferedImage(realImages[0]), "real_image_dcgan_with_noise.png");
            }
        }
    }

    private double[][] addNoise(double[][] image_array, double scale) {

        for (int i = 0; i < image_array.length; i++) {
            for (int j = 0; j < image_array[i].length; j++) {
                image_array[i][j] = clamp(image_array[i][j] + Math.random() * scale, -1, 1);
            }
        }
        return image_array;
    }


    public void train_discriminator(double[][][][] realImages, double[][][][] fakeImages) {
        // train on real images
        train_discriminator(realImages, expected_real_output_disc);

        // train with fake images
        train_discriminator(fakeImages, expected_fake_output_disc);
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

        double[][] disc_fake_output_gradients = generatorLossGradient(disc_fake_outputs);

        double[][][][] disc_input_gradients_gen_output_gradients = discriminator.backwardBatch(disc_fake_output_gradients);

        // input to the discriminator is the output of the generator
        generator.updateParametersBatch(disc_input_gradients_gen_output_gradients);

        // for debugging
//        if(accuracy > 0.99)
        MiscUtils.saveImage(getBufferedImage(scaleMinMax(disc_input_gradients_gen_output_gradients[0][0])), "disc_gradient_wrt_input.png");
    }

    private void calculateModelMetrics() {
        double[] fake_outputs = new double[test_size];
        double[] real_outputs = new double[test_size];
        double num_correct_predictions = 0.0;
        int test_index = train_size; // we want it to test outside the train dataset
        for (int img_idx = 0; img_idx < test_size; img_idx++) {
            real_outputs[img_idx] = discriminator.getOutput(new double[][][]{zeroToOneToMinusOneToOne(img_to_mat(mnist_load_index(label, test_index++)))})[0];
            fake_outputs[img_idx] = discriminator.getOutput(generator.generateImage())[0];
            num_correct_predictions += (fake_outputs[img_idx] <= 0.5 ? 1 : 0) + (real_outputs[img_idx] > 0.5 ? 1 : 0);
        }
        accuracy = num_correct_predictions / (test_size * 2.0);

        disc_loss = discriminatorLoss(real_outputs, fake_outputs);
        gen_loss = generatorLoss(fake_outputs);
    }

    public double generatorLoss(double[] fake_outputs) {
        /**
         * source : https://neptune.ai/blog/gan-loss-functions
         * */
        double loss = 0.0;
        for (int i = 0; i < fake_outputs.length; i++)
            loss += Math.log(1 - fake_outputs[i] + epsilon);

        return loss / fake_outputs.length; // we want to minimize this function
    }

    public double[] generatorLossGradient(double[] fake_output) {
        // here we are assuming that the output of the discriminator is a 1d array with 1 value
        double[] gradient = new double[fake_output.length];
        gradient[0] = -1 / ((1 - fake_output[0]) + epsilon);
        return gradient;
    }

    public double[][] generatorLossGradient(double[][] fake_outputs_batch) {
        // each output from the discriminator is a 1d array with 1 value
        double[][] gradients = new double[fake_outputs_batch.length][];
        for (int i = 0; i < fake_outputs_batch.length; i++)
            gradients[i] = generatorLossGradient(fake_outputs_batch[i]);

        return gradients;
    }

    public double discriminatorLoss(double[] real_outputs, double[] fake_outputs) {
        double[] ones = new double[real_outputs.length];
        double[] zeros = new double[real_outputs.length];
        Arrays.fill(ones, 1);

        double real_loss = lossBinaryCrossEntropy(real_outputs, ones);
        double fake_loss = lossBinaryCrossEntropy(fake_outputs, zeros);

        return real_loss + fake_loss;
    }

    private static final double epsilon = 1e-6;

}
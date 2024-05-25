package DCGAN;

import DCGAN.networks.Critic;
import DCGAN.networks.Generator_Implementation_Without_Batchnorm;
import DCGAN.util.MiscUtils;

import java.util.Arrays;
import java.util.Random;
import java.util.logging.Logger;

import static DCGAN.util.MiscUtils.*;


public class WGAN {
    final private static Logger logger = Logger.getLogger(WGAN.class.getName());

    int train_size = 1000;
    int test_size = 10;
    int label = 9;
    double learning_rate_gen = 0.0003;
    double learning_rate_disc = 0.0001;
    int batch_size = 8; // 1 for sgd

    private double disc_loss, gen_loss;

    Critic critic = new Critic(batch_size, learning_rate_disc);
    Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(batch_size, learning_rate_gen);

    double[] expected_real_output_disc = {1};
    double[] expected_fake_output_disc = {0};

    public static void main(String[] args) {
        WGAN wgan = new WGAN();
        wgan.train();
    }

    public void train() {
        generator.verbose = true;
        critic.verbose = true;

        // minibatch gradient descent
        for (int epochs = 0; epochs < 1000000; epochs++) {
            int index = 0;// reset our index to 0
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {
                // Load images
                double[][][][] fakeImages = generator.forwardBatch();
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++)
                    realImages[real_idx] = new double[][][]{
                            addNoise(
                                    zeroToOneToMinusOneToOne(img_to_mat(mnist_load_index(label, index++))),
                                    0)
                    };

                // train the critic
                double[][] disc_fake_outputs = critic.forwardBatch(fakeImages);
                train_critic_fake(disc_fake_outputs);

                double[][] disc_real_outputs = critic.forwardBatch(realImages);
                train_critic_real(disc_real_outputs);

                // train generator
                train_generator(critic.forwardBatch(fakeImages));

                // generate image
                double[][][] gen_img = generator.generateImage();
                saveImage(getBufferedImage(gen_img), "generated_image_wgan.png");

                // calculating and displaying our model metrics
                calculateModelMetrics();
                logger.info("Epoch : " + epochs + " batch:" + batch_idx);
                logger.info("Gen_Loss : " + gen_loss);
                logger.info("Disc_Loss : " + disc_loss);

                MiscUtils.saveImage(getBufferedImage(realImages[0]), "real_image_dcgan_with_noise.png");

            }
        }
    }

    private double[][] addNoise(double[][] image_array, double scale) {

        for (int i = 0; i < image_array.length; i++) {
            for (int j = 0; j < image_array[i].length; j++) {
                image_array[i][j] = clamp(image_array[i][j] + (Math.random() - 0.5) * 2 * scale, -1, 1);
            }
        }
        return image_array;
    }

    public void train_critic_fake(double[][] fake_outputs) {
        double[][] disc_output_gradients = new double[batch_size][1];
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            disc_output_gradients[img_idx][0] = -1.0 / fake_outputs.length;
        }

        critic.updateParametersBatch(disc_output_gradients);

        // for debugging
        System.out.println("Critic output for fake : " + Arrays.toString(fake_outputs[0]));
    }

    public void train_critic_real(double[][] real_outputs) {
        double[][] disc_output_gradients = new double[batch_size][1];
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            disc_output_gradients[img_idx][0] = +1.0 / real_outputs.length;
        }

        critic.updateParametersBatch(disc_output_gradients);

        // for debugging
        System.out.println("Critic output for real : " + Arrays.toString(real_outputs[0]));
    }

    public void train_generator(double[][] disc_fake_outputs) {
        double[][] disc_fake_output_gradients = new double[disc_fake_outputs.length][1];
        for (int sample_idx = 0; sample_idx < disc_fake_outputs.length; sample_idx++)
            disc_fake_output_gradients[sample_idx][0] = +1.0 / disc_fake_outputs.length;

        double[][][][] disc_input_gradients_gen_output_gradients = critic.backwardBatch(disc_fake_output_gradients);

        // input to the discriminator is the output of the generator
        generator.updateParametersBatch(disc_input_gradients_gen_output_gradients);

        // for debugging
        MiscUtils.saveImage(getBufferedImage(scaleMinMax(disc_input_gradients_gen_output_gradients[0][0])), "critic_in_gradient_wrt_input.png");
    }

    private void calculateModelMetrics() {
        double[] fake_outputs = new double[test_size];
        double[] real_outputs = new double[test_size];
        int test_index = train_size; // train_size; // we want it to test outside the train dataset
        for (int img_idx = 0; img_idx < test_size; img_idx++) {
            real_outputs[img_idx] = critic.getOutput(new double[][][]{zeroToOneToMinusOneToOne(img_to_mat(mnist_load_index(label, test_index++)))})[0];
            fake_outputs[img_idx] = critic.getOutput(generator.generateImage())[0];
        }

        disc_loss = criticLoss(real_outputs, fake_outputs);
        gen_loss = generatorLoss(fake_outputs);
    }

    public static double criticLoss(double[] real_outputs, double[] fake_outputs) {
        double avg_real_output = MiscUtils.mean(real_outputs), avg_fake_output = MiscUtils.mean(fake_outputs);
        return -(avg_real_output - avg_fake_output); // we want to minimize this
    }


    public static double generatorLoss(double[] fake_outputs) {
        double avg_fake_output = MiscUtils.mean(fake_outputs);
        return -avg_fake_output/fake_outputs.length;
    }
}
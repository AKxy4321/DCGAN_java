package DCGAN;

import DCGAN.networks.Critic_Spectral_Norm;
import DCGAN.networks.Generator_Implementation_Without_Batchnorm;
import DCGAN.util.MiscUtils;

import java.util.logging.Logger;

import static DCGAN.util.MiscUtils.*;


public class WGAN_With_SpectralNorm {
    final private static Logger logger = Logger.getLogger(WGAN_With_SpectralNorm.class.getName());

    int train_size = 1000;
    int test_size = 5;
    int label = 3;
    double learning_rate_gen = 0.0005;
    double learning_rate_critic = 0.0005;

    private int n_critics = 5;

    int batch_size = 32; // 1 for sgd

    private double disc_loss, gen_loss;

    Generator_Implementation_Without_Batchnorm generator = new Generator_Implementation_Without_Batchnorm(batch_size, learning_rate_gen);
    Critic_Spectral_Norm critic = new Critic_Spectral_Norm(batch_size, learning_rate_critic);

    public static void main(String[] args) {
        WGAN_With_SpectralNorm wgan = new WGAN_With_SpectralNorm();
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

                for (int i = 0; i < n_critics; i++) {
                    train_critic(realImages, fakeImages);
                }

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

                MiscUtils.saveImage(getBufferedImage(realImages[0]), "real_image_wgan_with_noise.png");

            }
        }
    }


    public static double criticLoss(double[] real_outputs, double[] fake_outputs) {
        double avg_real_output = MiscUtils.mean(real_outputs), avg_fake_output = MiscUtils.mean(fake_outputs);
        return -(avg_real_output - avg_fake_output); // we want to minimize this
    }


    public static double generatorLoss(double[] fake_outputs) {
        double avg_fake_output = MiscUtils.mean(fake_outputs);
        return -avg_fake_output / fake_outputs.length;
    }

    private void train_critic(double[][][][] realImages, double[][][][] fakeImages) {
        // train on real
        double[][] real_outputs = critic.forwardBatch(realImages).clone();
        double[][] disc_real_output_gradients = new double[batch_size][1];
        for (int img_idx = 0; img_idx < batch_size; img_idx++)
            disc_real_output_gradients[img_idx][0] = -1.0 / batch_size; // gradient for output of real images
        critic.updateParametersBatch(disc_real_output_gradients);

        // train on fake
        double[][] fake_outputs = critic.forwardBatch(fakeImages).clone();
        double[][] disc_fake_output_gradients = new double[batch_size][1];
        for (int img_idx = 0; img_idx < batch_size; img_idx++)
            disc_fake_output_gradients[img_idx][0] = +1.0 / batch_size; // gradient for output of fake images
        critic.updateParametersBatch(disc_fake_output_gradients);

        // for debugging
        System.err.printf("Critic output for real : %f, fake : %f\n", real_outputs[0][0], fake_outputs[0][0]);

        // reshaping our outputs array
        double[] fake_outputs_1d = new double[fake_outputs.length];
        double[] real_outputs_1d = new double[fake_outputs.length];
        for (int i = 0; i < fake_outputs_1d.length; i++) {
            fake_outputs_1d[i] = fake_outputs[i][0];
            real_outputs_1d[i] = real_outputs[i][0];
        }
        System.err.printf("Training loss for critic : %f", criticLoss(real_outputs_1d, fake_outputs_1d));
    }

    public void train_generator(double[][] disc_fake_outputs) {
        double[][] disc_fake_output_gradients = new double[disc_fake_outputs.length][1];
        for (int sample_idx = 0; sample_idx < disc_fake_outputs.length; sample_idx++)
            disc_fake_output_gradients[sample_idx][0] = -1.0 / disc_fake_outputs.length;

        double[][][][] disc_input_gradients_gen_output_gradients = critic.backwardBatch(disc_fake_output_gradients);

        // input to the discriminator is the output of the generator
        generator.updateParametersBatch(disc_input_gradients_gen_output_gradients);

        // for debugging
        MiscUtils.saveImage(getBufferedImage(scaleMinMax(disc_input_gradients_gen_output_gradients[0][0])), "critic_in_gradient_wrt_input.png");
        MiscUtils.prettyprint(disc_input_gradients_gen_output_gradients[0][0]);
    }

    private double[][] addNoise(double[][] image_array, double scale) {

        for (int i = 0; i < image_array.length; i++) {
            for (int j = 0; j < image_array[i].length; j++) {
                image_array[i][j] = clamp(image_array[i][j] + (Math.random() - 0.5) * 2 * scale, -1, 1);
            }
        }
        return image_array;
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
}

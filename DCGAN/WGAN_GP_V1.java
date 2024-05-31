package DCGAN;

import DCGAN.networks.Critic;
import DCGAN.networks.Critic_GP;
import DCGAN.networks.Generator_Implementation_Without_Batchnorm;
import DCGAN.optimizers.AdamHyperparameters;
import DCGAN.optimizers.OptimizerHyperparameters;
import DCGAN.util.MathUtils;
import DCGAN.util.MiscUtils;

import java.io.Serializable;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.logging.Logger;

import static DCGAN.util.MathUtils.mean;
import static DCGAN.util.MiscUtils.*;
import static DCGAN.util.SerializationUtils.loadObject;
import static DCGAN.util.SerializationUtils.saveObject;


public class WGAN_GP_V1 implements Serializable {
    private static final long serialVersionUID = 1L;
    final private static Logger logger = Logger.getLogger(WGAN_GP_V1.class.getName());

    int train_size = 1500;
    int test_size = 5;
    int label = 9;
    double learning_rate_gen = 2e-4;
    double learning_rate_critic = 2e-4;

    private int n_critics = 7;

    int batch_size = 32; // 1 for sgd

    double lambda = 10; // controls the strength of gradient penalty
    private double disc_loss, gen_loss;

    Generator_Implementation_Without_Batchnorm generator;
    Critic_GP critic;

    OptimizerHyperparameters generator_opt = new AdamHyperparameters(learning_rate_gen, 0, 0.9, 1e-4);
    OptimizerHyperparameters critic_opt = new AdamHyperparameters(learning_rate_critic, 0, 0.9, 1e-4);

    public static void main(String[] args) {
        WGAN_GP_V1 wgan = new WGAN_GP_V1();
        wgan.train();
    }

    public WGAN_GP_V1() {
        generator = (Generator_Implementation_Without_Batchnorm) loadObject("models/generator_wgan_gp_no_batchnorm_intermediate.ser");
        critic = (Critic_GP) loadObject("models/critic_wgan_gp_intermediate.ser");

        if (generator == null)
            generator = new Generator_Implementation_Without_Batchnorm(batch_size, generator_opt);
        if (critic == null)
            critic = new Critic_GP(batch_size, critic_opt);

        if (generator.getOptimizerHyperparameters() != generator_opt)
            generator.setOptimizerHyperparameters(generator_opt);
        if (critic.getOptimizerHyperparameters() != critic_opt)
            critic.setOptimizerHyperparameters(critic_opt);

        critic.setClip(-10000000, 1000000);

        generator.verbose = true;
        critic.verbose = true;
    }

    public void train() {
        LocalDateTime startTime = LocalDateTime.now();
        // minibatch gradient descent
        for (int epoch = 0; epoch < 1000000; epoch++) {
            int index = 0;// reset our index to 0
            for (int batch_idx = 0; batch_idx < train_size / batch_size; batch_idx++) {
                // Load images
                double[][][][] fakeImages = generator.forwardBatch();
                double[][][][] realImages = new double[batch_size][1][28][28];
                for (int real_idx = 0; real_idx < batch_size; real_idx++)
                    realImages[real_idx] = new double[][][]{
                            zeroToOneToMinusOneToOne(img_to_mat(mnist_load_index(label, index++))),
                    };

                for (int i = 0; i < n_critics; i++) {
                    train_critic(realImages, fakeImages);
                }

                // train generator
                train_generator(critic.forwardBatch(fakeImages));

                // generate image
                double[][][] gen_img = generator.generateImage();
                saveImage(getBufferedImage(gen_img), "outputs/generated_image_wgan.png");

                // calculating and displaying our model metrics
                calculateModelMetrics();
                logger.info("Epoch : " + epoch + " batch:" + batch_idx);
                logger.info("Gen_Loss : " + gen_loss);
                logger.info("Disc_Loss : " + disc_loss);

                MiscUtils.saveImage(getBufferedImage(realImages[0]), "outputs/real_image_wgan.png");
                LocalDateTime currentTime = LocalDateTime.now();
                if (Duration.between(startTime, currentTime).toMinutes() > 5) {
                    startTime = currentTime;
                    saveObject(generator, "models/generator_wgan_gp_no_batchnorm.ser");
                    saveObject(critic, "models/critic_gp_wgan.ser");
                }
            }
        }
    }


    public static double criticLoss(double[] real_outputs, double[] fake_outputs) {
        double avg_real_output = mean(real_outputs), avg_fake_output = mean(fake_outputs);
        return -(avg_real_output - avg_fake_output); // we want to minimize this
    }


    public static double generatorLoss(double[] fake_outputs) {
        double avg_fake_output = mean(fake_outputs);
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

        // gradient penalty
        double[] gradient_penalty = new double[batch_size];

        double[][][][] interpolatedImages = new double[batch_size][1][28][28];
        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            double mixing_factor = Math.random();
            // storing our interpolated image
            // interpolated image = real image * mixing_factor + fake image * (1 - mixing_factor)
            addScaledArrays(interpolatedImages[img_idx + batch_size * 2], // destination array where the result will be stored
                    realImages[img_idx], mixing_factor,
                    fakeImages[img_idx], 1 - mixing_factor);
        }
        double[][] outputs_for_interpolated = critic.forwardBatch(interpolatedImages);
        // fill this ones array with 1s
        double[][] ones = new double[batch_size][1];
        for (int i = 0; i < batch_size; i++) {
            ones[i][0] = 1;
        }
        // dou D(x_interpolated) / dou x_interpolated
        double[][][][] gradients_wrt_interpolated = critic.backwardBatch(ones);

        double[] l2_norm_interpolated_input_gradient = new double[batch_size];

        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            l2_norm_interpolated_input_gradient[img_idx] = MathUtils.l2Norm(gradients_wrt_interpolated[img_idx]);
            gradient_penalty[img_idx] = Math.pow(l2_norm_interpolated_input_gradient[img_idx] - 1, 2);
        }

        // dou GP/ dou x_interpolated
        double[][][][] dGPdInterpolated = new double[batch_size][1][interpolatedImages[0][0].length][interpolatedImages[0][0][0].length];

        for (int img_idx = 0; img_idx < batch_size; img_idx++) {
            for (int z = 0; z < interpolatedImages[0].length; z++) {
                for (int y = 0; y < interpolatedImages[0][0].length; y++) {
                    for (int x = 0; x < interpolatedImages[0][0][0].length; x++) {
                        dGPdInterpolated[img_idx][z][y][x] = (2.0 / batch_size) * lambda
                                * (l2_norm_interpolated_input_gradient[img_idx] - 1)
                                * gradients_wrt_interpolated[img_idx][z][y][x] / l2_norm_interpolated_input_gradient[img_idx];
                    }
                }
            }
        }

        critic.updateParametersBatch(ones, dGPdInterpolated);

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
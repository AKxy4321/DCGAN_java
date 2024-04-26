package DCGAN;

import java.util.Arrays;

public class DigitClassifier {
    public static void main(String[] args) {
        DigitClassifier dc = new DigitClassifier();
        dc.discriminator_execute();
    }

    public void discriminator_execute() {
        Discriminator_Implementation discriminator = new Discriminator_Implementation();
        int num_images = 100;
        int num_images_test = 100;
        int num_epochs = 1000;
        double learning_rate = 0.005;
        int batch_size = 8;

        double[][][] fakeImages_train = new double[num_images][28][28];
        double[][][] realImages_train = new double[num_images][28][28];

        double[][][] fakeImages_test = new double[num_images_test][28][28];
        double[][][] realImages_test = new double[num_images_test][28][28];

        for (int i = 0; i < num_images; i++) {
            System.out.print(i + " ");
            System.out.println();
            // so our images in mnist are in the range of 0 to 255, we convert it to range 0 to 1,
            // But we want the range to be between -1 to +1, so we convert it to that range
            //
            // problem 2: but the problem is that the value is still so small, the output at some of the layers is 0 every time you do forward prop
            // so we need to multiply it by a value to light up the neurons in the later layers
            realImages_train[i] =
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i)));
            fakeImages_train[i] =
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(7, i)));
        }

        for (int i = 0; i < num_images_test; i++) {
            realImages_test[i] = //DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i + num_images));
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i + num_images)));
            fakeImages_test[i] = //DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(0, i + num_images));
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(7, i + num_images)));
        }


        // Train discriminator to identify fake from real images using batch gradient descent
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double total_loss = 0.0;
            double[][] batchGradients = new double[num_images][1];

            // Train discriminator to identify fake from real images using batch gradient descent
            for (int i = 0; i < num_images; i++) {
                double[][] image = epoch % 2 == 0 ? realImages_train[i] : fakeImages_train[i];
                double loss = lossDiscriminatorMSE(discriminator.getOutput(image), new double[]{epoch % 2 == 0 ? 1 : 0});
                total_loss += loss;
                double[] output_gradient = new double[]{gradientDiscriminatorMSE(discriminator.getOutput(image), new double[]{epoch % 2 == 0 ? 1 : 0})};
                batchGradients[i] = output_gradient;

//                discriminator.updateParameters(output_gradient, learning_rate); // SGD
            }

            double[] mean_batch_gradient = UTIL.mean_1st_layer(batchGradients);

//            discriminator.backward(mean_batch_gradient);
            discriminator.updateParameters(mean_batch_gradient, learning_rate);

            // Calculate test loss and accuracy
            double test_loss = 0.0;
            double accuracy = 0.0;
            for (int i = 0; i < num_images_test; i++) {
                double[] test_real_outputs = discriminator.getOutput(realImages_test[i]);
                double[] test_fake_outputs = discriminator.getOutput(fakeImages_test[i]);
                test_loss += lossDiscriminatorMSE(new double[]{test_real_outputs[0], test_fake_outputs[0]}, new double[]{1, 0});
                accuracy += calculateAccuracy(test_real_outputs, test_fake_outputs);
                if (epoch == num_epochs - 1) {
                    System.out.println("Real output: " + Arrays.toString(test_real_outputs) + ", Fake output: " + Arrays.toString(test_fake_outputs));
                }
            }
            test_loss /= num_images_test;
            accuracy /= (2 * num_images_test);

            System.out.println("Epoch: " + (epoch + 1) + ", Average Training Loss: " + (total_loss / num_images) +
                    ", Test Loss: " + test_loss + ", Test Accuracy: " + accuracy + ", Total training loss: " + total_loss);
        }
    }

    public double lossDiscriminatorMSE(double[] outputs, double[] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return (loss / outputs.length);
    }
    public double lossDiscriminatorRMSE(double[] outputs, double[] expectedOutputs) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += Math.pow(outputs[i] - expectedOutputs[i], 2);
        }
        return Math.sqrt(loss / outputs.length);
    }

    public double gradientDiscriminatorRMSE(double[] outputs, double[] expectedOutputs) {
        double sum_squares = 0;
        double sum_errors = 0;
        int n= outputs.length;
        for (int i = 0; i < n; i++) {
            sum_squares += Math.pow(outputs[i] - expectedOutputs[i], 2);

            sum_errors += (outputs[i] - expectedOutputs[i]);
        }
        double mse = sum_squares / n;
        double rmse = Math.sqrt(mse);
        double gradient = -(1/rmse)*sum_errors/n;
        return gradient;
    }
    public double gradientDiscriminatorMSE(double[] outputs, double[] expectedOutputs) {
        double gradient = 0;
        for (int i = 0; i < outputs.length; i++) {
            gradient += 2 * (outputs[i] - expectedOutputs[i]);
        }
        return gradient / outputs.length;
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
}

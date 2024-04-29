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
        double learning_rate = 5*1e-4;
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
            realImages_train[i] = UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(3, i)));
            fakeImages_train[i] = UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(8, i)));
        }

        for (int i = 0; i < num_images_test; i++) {
            realImages_test[i] = //DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(1, i + num_images));
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(3, i + num_images)));
            fakeImages_test[i] = //DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(0, i + num_images));
                    UTIL.zeroToOneToMinusOneToOne(DCGAN.UTIL.img_to_mat(DCGAN.UTIL.mnist_load_index(8, i + num_images)));
        }


        // Train discriminator to identify fake from real images using batch gradient descent
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            double total_loss = 0.0;
            int[] indices = new int[num_images];
            for (int i = 0; i < num_images; i++) {
                indices[i] = i;
            }
            shuffle(indices);

            // Train discriminator using minibatch gradient descent
            for (int i = 0; i < num_images; i += batch_size) {
                int endIndex = Math.min(i + batch_size, num_images);
                double[][][] batch_real_images = new double[endIndex - i][28][28];
                double[][][] batch_fake_images = new double[endIndex - i][28][28];
                double[][] batchGradients = new double[endIndex - i][1];

                // Load minibatch of real and fake images
                for (int j = i; j < endIndex; j++) {
                    batch_real_images[j - i] = realImages_train[indices[j]];
                    batch_fake_images[j - i] = fakeImages_train[indices[j]];
                }

                // Compute loss and gradients for the minibatch
                for (int j = 0; j < endIndex - i; j++) {
                    double[][] image = epoch % 2 == 0 ? batch_real_images[j] : batch_fake_images[j];
                    double loss = lossBinaryCrossEntropy(discriminator.getOutput(image), new double[]{epoch % 2 == 0 ? 1 : 0});
                    total_loss += loss;
                    double[] output_gradient = gradientBinaryCrossEntropy(discriminator.getOutput(image), new double[]{epoch % 2 == 0 ? 1 : 0});
                    batchGradients[j] = output_gradient;
                }

                // Update discriminator parameters using mean gradients of the minibatch
                double[] mean_batch_gradient = UTIL.mean_1st_layer(batchGradients);
                discriminator.updateParameters(mean_batch_gradient, learning_rate);
            }

            // Calculate test loss and accuracy
            double test_loss = 0.0;
            double accuracy = 0.0;
            for (int i = 0; i < num_images_test; i++) {
                double[] test_real_outputs = discriminator.getOutput(realImages_test[i]);
                double[] test_fake_outputs = discriminator.getOutput(fakeImages_test[i]);
                test_loss += lossBinaryCrossEntropy(new double[]{test_real_outputs[0], test_fake_outputs[0]}, new double[]{1, 0});
                accuracy += calculateAccuracy(test_real_outputs, test_fake_outputs);
                if (epoch == num_epochs - 1) {
                    System.out.println("Real output: " + Arrays.toString(test_real_outputs) + ", Fake output: " + Arrays.toString(test_fake_outputs));
                }
            }
            test_loss /= num_images_test;
            accuracy /= (2 * num_images_test);

            System.out.println("Epoch: " + (epoch + 1) + ", Average Training Loss: " + (total_loss / num_images) + ", Test Loss: " + test_loss + ", Test Accuracy: " + accuracy + ", Total training loss: " + total_loss);
        }
    }

    private void shuffle(int[] indices) {
        for (int i = 0; i < indices.length; i++) {
            int randomIndex = (int) (Math.random() * indices.length);
            int temp = indices[i];
            indices[i] = indices[randomIndex];
            indices[randomIndex] = temp;
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
        int n = outputs.length;
        for (int i = 0; i < n; i++) {
            sum_squares += Math.pow(outputs[i] - expectedOutputs[i], 2);

            sum_errors += (outputs[i] - expectedOutputs[i]);
        }
        double mse = sum_squares / n;
        double rmse = Math.sqrt(mse);
        double gradient = -(1 / rmse) * sum_errors / n;
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

    public double lossBinaryCrossEntropy(double[] outputs, double[] labels) {
        double loss = 0;
        for (int i = 0; i < outputs.length; i++) {
            loss += labels[i] * Math.log(outputs[i]) + (1 - labels[i]) * Math.log(1 - outputs[i]);
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

    public static double epsilon = 0.00001;

}
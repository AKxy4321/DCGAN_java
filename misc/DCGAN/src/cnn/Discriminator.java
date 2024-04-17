package cnn;

import UTIL.Mat;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import javax.imageio.ImageIO;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

public class Discriminator {

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    public static void saveWeights(float[][][] filters, float[][][] filters2, float[][] weights, float[][] bias, float[][] weights2, float[][] bias2) throws IOException {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream("model_weights.dat"))) {
            outputStream.writeObject(filters);
            outputStream.writeObject(filters2);
            outputStream.writeObject(weights);
            outputStream.writeObject(bias);
            outputStream.writeObject(weights2);
            outputStream.writeObject(bias2);
        }
    }

    public static float[][] img_to_mat(BufferedImage imageToPixelate) {
        int w = imageToPixelate.getWidth(), h = imageToPixelate.getHeight();
        int[] pixels = imageToPixelate.getRGB(0, 0, w, h, null, 0, w);
        float[][] dta = new float[w][h];

        for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel++) {
            dta[row][col] = (((int) pixels[pixel] >> 16 & 0xff)) / 255.0f;
            col++;
            if (col == w) {
                col = 0;
                row++;
            }
        }
        return dta;
    }

    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "D:\\Projects\\ZirohLabs---DCGAN\\misc\\CNN\\data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        return load_image(final_path);
    }

    public static BufferedImage mnist_load_index(int label, int index) throws IOException {
        String mnist_path = "D:\\Projects\\ZirohLabs---DCGAN\\misc\\CNN\\data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        String final_path = mnist_path + "\\" + label + "\\" + files[index];
        return load_image(final_path);
    }

    public static float[][][] flatten(float[][][] input) {
        int batchSize = input.length;
        int height = input[0].length;
        int width = input[0][0].length;

        float[][][] flattened = new float[batchSize][1][height * width];

        for (int b = 0; b < batchSize; b++) {
            int index = 0;
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                        flattened[b][0][index++] = input[b][h][w];
                }
            }
        }

        return flattened;
    }

    float[][] fakeImage;
    public void takeFakeImage(float[][] img) {
        this.fakeImage = img;
    }

    public void train(int training_size) throws IOException {
        int label_counter = 0;
        float ce_loss=0;
        int correct_predictions_batch = 0;
        int total_predictions_batch = 0;
        float acc_sum=0.0f;
        float learn_rate=1e-6f;

        Convolution conv1 = new Convolution(64, 5, 1);
        ReLU relu1 = new ReLU();
        MaxPool pool1 = new MaxPool();

        Convolution conv2 = new Convolution(128, 5, 1);
        ReLU relu2 = new ReLU();
        MaxPool pool2 = new MaxPool();
        int flattenInputSize = 4 * 4 * 128;

        Dense dense = new Dense(flattenInputSize, 1, 128, 1);
        Sigmoid sigmoid = new Sigmoid(flattenInputSize, 1, 128, 1);

        int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        float[][] out_l = new float[1][10];
        for (int i = 0; i < training_size; i++) {
            BufferedImage bi = mnist_load_index(label_counter, index[label_counter]);
            int correct_label = label_counter;
            if (label_counter == 9) {
                label_counter = 0;
            } else {
                label_counter++;
            }

            index[label_counter]++;

            float [][] realImage = img_to_mat(bi);
            // System.out.println(Arrays.toString(realImage[0]));

            //Forward propagation
            float[][][] output = conv1.forward(realImage);
            output = relu1.forward(output);
            output = pool1.forward(output);

            output = conv2.forward(output);
            output = relu2.forward(output);
            output = pool2.forward(output);

            out_l = dense.forward(output);
            out_l = sigmoid.forward(out_l);

            ce_loss += (float) -Math.log(out_l[0][correct_label]);
            correct_predictions_batch += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            total_predictions_batch++;


            //Backward propagation
            float[][] gradient=Mat.v_zeros(10);
            gradient[0][correct_label]=-1/out_l[0][correct_label];
            float[][][] sigmoid_gradient=sigmoid.backprop(gradient,learn_rate);
            float[][][] dense_gradient=dense.backprop(sigmoid_gradient,learn_rate);
            float[][][] mp2_gradient=pool2.backprop(dense_gradient);
            float[][][] relu2_gradient = relu2.backprop(mp2_gradient);
            float[][][] conv2_gradient = conv2.backprop(relu2_gradient, learn_rate);
            float[][][] mp1_gradient = pool1.backprop(conv2_gradient);
            float[][][] relu1_gradient = pool1.backprop(mp1_gradient);
            float[][][] conv1_gradient = conv1.backprop(relu1_gradient,learn_rate);

            if(i % 100 == 99){
                float accuracy_batch = (float)correct_predictions_batch / total_predictions_batch * 100;
                System.out.println(" step: "+ i+ " loss: "+ce_loss/100.0+" accuracy: "+accuracy_batch+"%");
                ce_loss=0;
                acc_sum+=accuracy_batch;
            }
        }
        System.out.println("average accuracy:- "+acc_sum/training_size+"%");
        saveWeights(conv1.get_Filters(), conv2.get_Filters(), dense.weights, dense.bias ,sigmoid.weights, sigmoid.bias);
    }

   public static void main(String[] args) throws IOException {
       new Discriminator().train(1000);
   }
}
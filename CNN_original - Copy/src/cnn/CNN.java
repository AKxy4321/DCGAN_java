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

public class CNN {

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
    }

    public static void saveWeights(float[][][] filters, float[][] weights, float[][] bias) throws IOException {
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream("model_weights.dat"))) {
            outputStream.writeObject(filters);
            outputStream.writeObject(weights);
            outputStream.writeObject(bias);
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
        String mnist_path = "data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        return load_image(final_path);
    }

    public static BufferedImage mnist_load_index(int label, int index) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\training";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        String final_path = mnist_path + "\\" + label + "\\" + files[index];
        return load_image(final_path);
    }

    public static void train(int training_size) throws IOException {
        int label_counter = 0;
        float ce_loss=0;
        int correct_predictions_batch = 0;
        int total_predictions_batch = 0;
        float acc_sum=0.0f;
        float learn_rate=1e-6f;

        Convolution conv=new Convolution(64, 5, 1);
        ReLU relu = new ReLU();
        MaxPool pool=new MaxPool();
        SoftMax softmax=new SoftMax(12 * 12 * 64, 10, 64, 12);

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

            float[][] pxl = img_to_mat(bi);
            float[][][] out = conv.forward(pxl);

            out = relu.forward(out);
            out = pool.forward(out);
            out_l = softmax.forward(out);

            ce_loss += (float) -Math.log(out_l[0][correct_label]);
            correct_predictions_batch += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            total_predictions_batch++;

            float[][] gradient=Mat.v_zeros(10);
            gradient[0][correct_label]=-1/out_l[0][correct_label];
            float[][][] sm_gradient=softmax.backprop(gradient,learn_rate);
            float[][][] mp_gradient=pool.backprop(sm_gradient);
            float[][][] relu_gradient = relu.backprop(mp_gradient);
            conv.backprop(relu_gradient, learn_rate);
            if(i % 100 == 99){
                float accuracy_batch = (float)correct_predictions_batch / total_predictions_batch * 100;
                System.out.println(" step: "+ i+ " loss: "+ce_loss/100.0+" accuracy: "+accuracy_batch+"%");
                ce_loss=0;
                acc_sum+=accuracy_batch;
            }
        }
        System.out.println("average accuracy:- "+acc_sum/training_size+"%");
        saveWeights(conv.get_Filters(), softmax.weights, softmax.bias);
    }

    public static void main(String[] args) throws IOException {
        train(1000);
    }
}

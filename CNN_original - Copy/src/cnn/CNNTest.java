package cnn;

import UTIL.Mat;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Random;

public class CNNTest {

    public static BufferedImage load_image(String src) throws IOException {
        return ImageIO.read(new File(src));
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

    public static void test(int testing_size) throws IOException {
        int label_counter = 0;
        int correct_predictions_batch = 0;
        int total_predictions_batch = 0;

        float[][][] filters;
        float[][] weights;
        float[][] bias;

        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream("model_weights.dat"))) {
            filters = (float[][][]) inputStream.readObject();
            weights = (float[][]) inputStream.readObject();
            bias = (float[][]) inputStream.readObject();
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }

        Convolution conv = new Convolution(filters, 1);
        ReLU relu = new ReLU();
        MaxPool pool = new MaxPool();
        SoftMax softmax = new SoftMax(weights, bias);
        int[] index = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        for (int i = 0; i < testing_size; i++) {
            float[][] out_l = new float[1][10];
            BufferedImage bi = mnist_load_index(label_counter, index[label_counter]);
            int correct_label = label_counter;
            if(label_counter==9){
                label_counter=0;
            }else{
                label_counter++;
            }

            index[label_counter]++;

            float[][] pxl = img_to_mat(bi);
            float[][][] out = conv.forward(pxl);

            out = relu.forward(out);
            out = pool.forward(out);
            out_l = softmax.forward(out);

            correct_predictions_batch += correct_label == Mat.v_argmax(out_l) ? 1 : 0;
            total_predictions_batch++;

            if(i % 100 == 99){
                float accuracy_batch = (float)correct_predictions_batch / total_predictions_batch * 100;
                System.out.println(" step: "+ i+" accuracy: "+accuracy_batch+"%");
            }
        }

        float accuracy = (float) correct_predictions_batch / total_predictions_batch * 100;
        System.out.println("Accuracy on test set: " + accuracy + "%");
    }

    public static BufferedImage mnist_load_random(int label) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\testing";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        int random_index = new Random().nextInt(files.length);
        String final_path = mnist_path + "\\" + label + "\\" + files[random_index];
        return load_image(final_path);
    }

    public static BufferedImage mnist_load_index(int label, int index) throws IOException {
        String mnist_path = "data\\mnist_png\\mnist_png\\testing";
        File dir = new File(mnist_path + "\\" + label);
        String[] files = dir.list();
        assert files != null;
        String final_path = mnist_path + "\\" + label + "\\" + files[index];
        return load_image(final_path);
    }

    public static void main(String[] args) throws IOException {
        test(2000);
    }
}


package DCGAN;

import java.util.Random;

public class XavierInitializer {

    public static double[] xavierInit1D(int size) {
        double[] weights = new double[size];
        Random rand = new Random();
        double scale = Math.sqrt(1.0 / size);
        for (int i = 0; i < size; i++) {
            weights[i] = rand.nextGaussian() * scale;
        }
        return weights;
    }

    public static double[][] xavierInit2D(int rows, int cols) {
        double[][] weights = new double[rows][cols];
        Random rand = new Random();
        double scale = Math.sqrt(1.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = rand.nextGaussian() * scale;
            }
        }
        return weights;
    }

    public static double[][][] xavierInit3D(int depth, int height, int width) {
        double[][][] weights = new double[depth][height][width];
        Random rand = new Random();
        double scale = Math.sqrt(1.0 / (depth + height + width));
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    weights[i][j][k] = rand.nextGaussian() * scale;
                }
            }
        }
        return weights;
    }

    public static double[][][][] xavierInit4D(int numFilters, int numFiltersPrev, int filterSize) {
        double[][][][] filters = new double[numFilters][numFiltersPrev][filterSize][filterSize];
        Random rand = new Random();
        double scale = Math.sqrt(1.0 / (numFilters + numFiltersPrev + filterSize + filterSize));
        for (int i = 0; i < numFilters; i++) {
            for (int j = 0; j < numFiltersPrev; j++) {
                for (int k = 0; k < filterSize; k++) {
                    for (int l = 0; l < filterSize; l++) {
                        filters[i][j][k][l] = rand.nextGaussian() * scale;
                    }
                }
            }
        }
        return filters;
    }
}

package DCGAN.util;

import java.util.Random;

public class ArrayInitializer {

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

    public static double[][][][] initRandom4D(int x, int y, int z, int w) {
        double[][][][] array = new double[x][y][z][w];
        initRandom4D(array);
        return array;
    }

    public static void initRandom4D(double[][][][] dest) {
        Random rand = new Random();
        for (int i = 0; i < dest.length; i++) {
            for (int j = 0; j < dest[i].length; j++) {
                for (int k = 0; k < dest[i][j].length; k++) {
                    for (int l = 0; l < dest[i][j][k].length; l++) {
                        dest[i][j][k][l] = rand.nextGaussian();
                    }
                }
            }
        }
    }

    public static double[][][] initRandom3D(int x, int y, int z) {
        double[][][] array = new double[x][y][z];
        initRandom3D(array);
        return array;
    }

    public static void initRandom3D(double[][][] dest) {
        Random rand = new Random();
        for (int i = 0; i < dest.length; i++) {
            for (int j = 0; j < dest[i].length; j++) {
                for (int k = 0; k < dest[i][j].length; k++) {
                    dest[i][j][k] = rand.nextGaussian();
                }
            }
        }
    }

    public static double[] initGaussian1D(int x) {
        double[] array = new double[x];
        initGaussian1D(array);
        return array;
    }

    public static void initGaussian1D(double[] dest) {
        Random rand = new Random();
        for (int i = 0; i < dest.length; i++) {
            dest[i] = rand.nextGaussian();
        }
    }

    public static double[] sphericalNoiseVector(int noise_len, double radius) {
        double[] noise = new double[noise_len];
        sphericalNoiseVector(noise, radius);
        return noise;
    }

    public static void sphericalNoiseVector(double[] noise, double radius) {
        initGaussian1D(noise);
        double mag = Math.max(mag(noise), 0.0001);
        for (int i = 0; i < noise.length; i++) {
            noise[i] *= radius / mag;
        }
    }

    public static double mag(double[] vec) {
        double sum = 0;
        for (int i = 0; i < vec.length; i++) {
            sum += vec[i] * vec[i];
        }
        return Math.sqrt(sum);
    }
}

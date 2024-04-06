package Ziroh;

import java.util.Arrays;

public class Matrix {

    private final int rows;
    private final int cols;
    private final double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public void set(int i, int j, double value) {
        data[i][j] = value;
    }

    public double get(int i, int j) {
        return data[i][j];
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public Matrix multiply(int k) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] * k);
            }
        }
        return result;
    }

    public Matrix square() {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] * data[i][j]);
            }
        }
        return result;
    }

    public Matrix power(int exponent) {
        Matrix result = this;
        for (int i = 1; i < exponent; i++) {
            result = result.multiply(this);
        }
        return result;
    }

    private Matrix multiply(Matrix matrix) {
        if (this.cols != matrix.getRows()) {
            throw new IllegalArgumentException("Number of columns of the first matrix must be equal to the number of rows of the second matrix for multiplication.");
        }
    
        Matrix result = new Matrix(this.rows, matrix.getCols());
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < matrix.getCols(); j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * matrix.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        return result;
    }    

    public Matrix add(Matrix matrix) {
        if (this.rows != matrix.getRows() || this.cols != matrix.getCols()) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for addition.");
        }
    
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.data[i][j] + matrix.get(i, j));
            }
        }
        return result;
    }
    
    public Matrix subtract(Matrix matrix) {
        if (this.rows != matrix.getRows() || this.cols != matrix.getCols()) {
            throw new IllegalArgumentException("Matrices must have the same dimensions for subtraction.");
        }
    
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(i, j, this.data[i][j] - matrix.get(i, j));
            }
        }
        return result;
    }
    
    public Matrix root(int exponent) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, Math.pow(data[i][j], 1.0 / exponent));
            }
        }
        return result;
    }

    public double average() {
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
        }
        return sum / (rows * cols);
    }

    public static Matrix zeros(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, 0.0);
            }
        }
        return matrix;
    }

    public static Matrix ones(int rows, int cols) {
        Matrix matrix = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, 1.0);
            }
        }
        return matrix;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            sb.append(Arrays.toString(data[i])).append("\n");
        }
        return sb.toString();
    }

    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.set(j, i, this.data[i][j]);
            }
        }
        return result;
    }    

    public static void main(String[] args) {
        Matrix matrix = new Matrix(2, 2);
        matrix.set(0, 0, 1);
        matrix.set(0, 1, 2);
        matrix.set(1, 0, 3);
        matrix.set(1, 1, 4);

        System.out.println("Original Matrix:");
        System.out.println(matrix);

        System.out.println("Matrix multiplied by 2:");
        System.out.println(matrix.multiply(2));

        System.out.println("Squared Matrix:");
        System.out.println(matrix.square());

        System.out.println("Matrix raised to the power of 3:");
        System.out.println(matrix.power(2));

        System.out.println("Matrix square root:");
        System.out.println(matrix.root(2));

        System.out.println("Matrix of zeros (2x2):");
        System.out.println(Matrix.zeros(2, 2));

        System.out.println("Matrix of ones (3x3):");
        System.out.println(Matrix.ones(3, 3));

        System.out.println("Average of matrix elements:");
        System.out.println(matrix.average());

            // Create two matrices
        Matrix matrix1 = new Matrix(2, 2);
        matrix1.set(0, 0, 1);
        matrix1.set(0, 1, 2);
        matrix1.set(1, 0, 3);
        matrix1.set(1, 1, 4);

        Matrix matrix2 = new Matrix(2, 2);
        matrix2.set(0, 0, 5);
        matrix2.set(0, 1, 6);
        matrix2.set(1, 0, 7);
        matrix2.set(1, 1, 8);

        // Display original matrices
        System.out.println("Matrix 1:");
        System.out.println(matrix1);
        System.out.println("Matrix 2:");
        System.out.println(matrix2);

        // Perform matrix addition
        System.out.println("Matrix Addition:");
        System.out.println(matrix1.add(matrix2));

        // Perform matrix subtraction
        System.out.println("Matrix Subtraction:");
        System.out.println(matrix1.subtract(matrix2));

        Matrix matrix3 = new Matrix(2, 3);
        matrix3.set(0, 0, 1);
        matrix3.set(0, 1, 2);
        matrix3.set(0, 2, 3);
        matrix3.set(1, 0, 4);
        matrix3.set(1, 1, 5);
        matrix3.set(1, 2, 6);

        // Display original matrix
        System.out.println("Original Matrix:");
        System.out.println(matrix3);

        // Transpose the matrix
        System.out.println("Transposed Matrix:");
        System.out.println(matrix3.transpose());
        }
}
package Ziroh;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;

public class SampleImageDisplay {

    public static void displaySampleImage(List<ImageClass> dataset) {
        // Choose a random image from the dataset
        Random random = new Random();
        ImageClass randomImage = dataset.get(random.nextInt(dataset.size()));

        // Get the image and class name from the ImageClass object
        BufferedImage image = randomImage.getImage();
        String className = randomImage.getClassName();

        // Display the image and class name in a JFrame
        JFrame frame = new JFrame("Sample Image: " + className);
        JPanel panel = new JPanel(new BorderLayout());
        JLabel imageLabel = new JLabel(new ImageIcon(image));
        JLabel classLabel = new JLabel("Class: " + className);
        panel.add(imageLabel, BorderLayout.CENTER);
        panel.add(classLabel, BorderLayout.SOUTH);
        frame.getContentPane().add(panel);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        try {
            String datasetPath = "Ziroh/dataset/test"; // Specify the path to your dataset folder
            List<ImageClass> dataset = ImageDatasetLoader.loadDataset(datasetPath);
            
            // Display a sample image from the loaded dataset
            displaySampleImage(dataset);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

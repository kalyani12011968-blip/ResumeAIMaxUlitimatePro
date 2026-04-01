import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

public class ResumeUIANDUX {

    static ResumeAIMaxUlitimatePro.NeuralNet model;

    public static void main(String[] args) {

        try {
            java.util.List<ResumeAIMaxUlitimatePro.Data> dataset =
                    ResumeAIMaxUlitimatePro.loadCSV("resume_dataset.csv");

            ResumeAIMaxUlitimatePro.buildVocab(dataset);

            int inputSize = ResumeAIMaxUlitimatePro.vocab.size() + 9;
            model = new ResumeAIMaxUlitimatePro.NeuralNet(inputSize);

            for (int epoch = 0; epoch < 30; epoch++) {
                for (ResumeAIMaxUlitimatePro.Data d : dataset) {

                    double[] vec = ResumeAIMaxUlitimatePro.combine(
                            ResumeAIMaxUlitimatePro.tfidfVector(d.text),
                            ResumeAIMaxUlitimatePro.extraFeatures(d.text)
                    );

                    model.train(vec, d.label);
                }
            }

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(null, "Dataset error: " + ex.getMessage());
            return;
        }

        // ================= FRAME =================
        JFrame frame = new JFrame("AI Resume Analyzer");
        frame.setSize(800, 650);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // ================= MAIN PANEL (Gradient) =================
        JPanel mainPanel = new JPanel() {
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                Color c1 = new Color(58, 123, 213);
                Color c2 = new Color(0, 210, 255);
                GradientPaint gp = new GradientPaint(0, 0, c1, getWidth(), getHeight(), c2);
                g2d.setPaint(gp);
                g2d.fillRect(0, 0, getWidth(), getHeight());
            }
        };
        mainPanel.setLayout(new BorderLayout());

        // ================= HEADER =================
        JLabel header = new JLabel("AI Resume Analyzer", JLabel.CENTER);
        header.setFont(new Font("Arial", Font.BOLD, 28));
        header.setForeground(Color.WHITE);
        header.setBorder(BorderFactory.createEmptyBorder(20, 0, 20, 0));

        // Optional image (place image in same folder)
        JLabel imageLabel = new JLabel();
        try {
            ImageIcon icon = new ImageIcon("resume.png"); // add your image
            Image img = icon.getImage().getScaledInstance(80, 80, Image.SCALE_SMOOTH);
            imageLabel.setIcon(new ImageIcon(img));
        } catch (Exception e) {}

        JPanel topPanel = new JPanel(new BorderLayout());
        topPanel.setOpaque(false);
        topPanel.add(header, BorderLayout.CENTER);
        topPanel.add(imageLabel, BorderLayout.WEST);

        // ================= FORM PANEL =================
        JPanel formPanel = new JPanel();
        formPanel.setLayout(new GridLayout(6, 2, 10, 10));
        formPanel.setBackground(Color.WHITE);
        formPanel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JTextField nameField = new JTextField();
        JTextField skillsField = new JTextField();
        JTextField expField = new JTextField();
        JTextField projectField = new JTextField();
        JTextField eduField = new JTextField();
        JTextField certField = new JTextField();

        formPanel.add(new JLabel("Name:")); formPanel.add(nameField);
        formPanel.add(new JLabel("Skills:")); formPanel.add(skillsField);
        formPanel.add(new JLabel("Experience:")); formPanel.add(expField);
        formPanel.add(new JLabel("Projects:")); formPanel.add(projectField);
        formPanel.add(new JLabel("Education:")); formPanel.add(eduField);
        formPanel.add(new JLabel("Certifications:")); formPanel.add(certField);

        // ================= BUTTON =================
        JButton analyzeBtn = new JButton("Analyze Resume");
        analyzeBtn.setBackground(new Color(0, 153, 76));
        analyzeBtn.setForeground(Color.WHITE);
        analyzeBtn.setFont(new Font("Arial", Font.BOLD, 16));

        // ================= RESULT AREA =================
        JTextArea resultArea = new JTextArea(5, 20);
        resultArea.setEditable(false);
        resultArea.setFont(new Font("Arial", Font.BOLD, 16));
        resultArea.setBorder(BorderFactory.createTitledBorder("Result"));

        JScrollPane scroll = new JScrollPane(resultArea);

        // ================= CENTER PANEL =================
        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.setOpaque(false);
        centerPanel.add(formPanel, BorderLayout.CENTER);
        centerPanel.add(analyzeBtn, BorderLayout.SOUTH);

        // ================= ADD COMPONENTS =================
        mainPanel.add(topPanel, BorderLayout.NORTH);
        mainPanel.add(centerPanel, BorderLayout.CENTER);
        mainPanel.add(scroll, BorderLayout.SOUTH);

        frame.add(mainPanel);

        // ================= BUTTON ACTION =================
        analyzeBtn.addActionListener(e -> {

            String text = nameField.getText() + " " +
                          skillsField.getText() + " " +
                          expField.getText() + " " +
                          projectField.getText() + " " +
                          eduField.getText() + " " +
                          certField.getText();

            double[] vec = ResumeAIMaxUlitimatePro.combine(
                    ResumeAIMaxUlitimatePro.tfidfVector(text),
                    ResumeAIMaxUlitimatePro.extraFeatures(text)
            );

            double score = model.forward(vec) * 100;

            String decision;
            if (score >= 75) decision = "SELECTED ";
            else if (score >= 60) decision = "MAYBE ";
            else decision = "REJECTED ";

            resultArea.setText("Result: " + decision);

            if (decision.contains("SELECTED"))
                resultArea.setForeground(new Color(0, 128, 0));
            else if (decision.contains("MAYBE"))
                resultArea.setForeground(Color.ORANGE);
            else
                resultArea.setForeground(Color.RED);
        });

        frame.setVisible(true);
    }
}

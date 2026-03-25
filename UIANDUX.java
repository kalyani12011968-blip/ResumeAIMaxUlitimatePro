import javax.swing.*;
import java.awt.*;
public class UIANDUX {
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
            JOptionPane.showMessageDialog(null,
                    "Error loading dataset: " + ex.getMessage());
            return;
        }
        JFrame frame = new JFrame("AI Resume Analyzer");
        frame.setSize(700, 600);
        frame.setLayout(new GridLayout(9, 2, 10, 10));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JTextField nameField = new JTextField();
        JTextField skillsField = new JTextField();
        JTextField expField = new JTextField();
        JTextField projectField = new JTextField();
        JTextField eduField = new JTextField();
        JTextField certField = new JTextField();
        JTextArea resultArea = new JTextArea();
        resultArea.setEditable(false);
        JButton analyzeBtn = new JButton("Analyze Resume");
        frame.add(new JLabel("Name:")); frame.add(nameField);
        frame.add(new JLabel("Skills:")); frame.add(skillsField);
        frame.add(new JLabel("Experience:")); frame.add(expField);
        frame.add(new JLabel("Projects:")); frame.add(projectField);
        frame.add(new JLabel("Education:")); frame.add(eduField);
        frame.add(new JLabel("Certifications:")); frame.add(certField);
        frame.add(analyzeBtn);
        frame.add(new JLabel("Result:"));
        frame.add(new JScrollPane(resultArea));
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
        });
        frame.setVisible(true);
    }
}

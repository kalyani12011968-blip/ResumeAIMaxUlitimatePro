import java.io.*;
import java.util.*;

public class ResumeAIMaxUlitimatePro {

    static int HIDDEN_SIZE = 256;
    static double LEARNING_RATE = 0.005;
    static int EPOCHS = 200;

    static Random rand = new Random();

    static Map<String, Integer> vocab = new HashMap<>();
    static Map<String, Double> idf = new HashMap<>();

    // TOKENIZE
    static List<String> tokenize(String text) {
        text = text.toLowerCase().replaceAll("[^a-z ]", "");
        return Arrays.asList(text.split("\\s+"));
    }

    // BUILD VOCAB
    static void buildVocab(List<Data> dataset) {
        int index = 0;
        Map<String, Integer> docCount = new HashMap<>();

        for (Data d : dataset) {
            Set<String> seen = new HashSet<>(tokenize(d.text));
            for (String word : seen) {
                docCount.put(word, docCount.getOrDefault(word, 0) + 1);
                if (!vocab.containsKey(word)) vocab.put(word, index++);
            }
        }

        int N = dataset.size();
        for (String word : vocab.keySet()) {
            int df = docCount.getOrDefault(word, 1);
            idf.put(word, Math.log((double) N / df));
        }
    }

    // TF-IDF
    static double[] tfidfVector(String text) {
        List<String> tokens = tokenize(text);
        double[] vec = new double[vocab.size()];

        Map<String, Integer> tf = new HashMap<>();
        for (String t : tokens)
            tf.put(t, tf.getOrDefault(t, 0) + 1);

        for (String word : tf.keySet()) {
            if (!vocab.containsKey(word)) continue;

            int index = vocab.get(word);
            double tfVal = 1 + Math.log(tf.get(word));
            double idfVal = idf.getOrDefault(word, 0.0);

            vec[index] = tfVal * idfVal;
        }

        double norm = 0;
        for (double v : vec) norm += v * v;
        norm = Math.sqrt(norm) + 1e-6;

        for (int i = 0; i < vec.length; i++)
            vec[i] /= norm;

        return vec;
    }

    // EXTRA FEATURES
    static double[] extraFeatures(String text) {
        text = text.toLowerCase();

        int skillCount = 0;
        if (text.contains("java")) skillCount += 4;
        if (text.contains("aws")) skillCount += 4;
        if (text.contains("docker")) skillCount += 3;
        if (text.contains("react")) skillCount += 1;
        if (text.contains("python")) skillCount += 2;
        if (text.contains("sql")) skillCount += 1;
        if (text.contains("spring")) skillCount += 3;

        if (skillCount > 5) skillCount += 2;

        int experience = 0;
        String[] words = text.split(" ");

        for (int i = 0; i < words.length; i++) {
            if (words[i].matches("\\d+") && i + 1 < words.length &&
                words[i + 1].contains("year")) {
                experience = Integer.parseInt(words[i]);
                break;
            }
        }

        double expLevel = experience > 5 ? 1 : 0.5;
        double project = text.contains("project") ? 1 : 0;
        int projectCount = text.split("project").length - 1;

        double education = 0;
        if (text.contains("bachelor")) education = 0.5;
        if (text.contains("master")) education = 0.8;
        if (text.contains("phd")) education = 1.0;

        double cert = text.contains("certified") ? 1 : 0;

        double role = 0;
        if (text.contains("developer")) role = 0.7;
        if (text.contains("engineer")) role = 0.8;
        if (text.contains("data scientist")) role = 1.0;

        double lengthFeature = Math.min(1.0, text.length() / 500.0);

        return new double[]{
            skillCount / 20.0,
            experience / 20.0,
            expLevel * 0.5,
            project * 0.5,
            projectCount / 10.0,
            education * 0.5,
            cert * 0.5,
            role * 0.5,
            lengthFeature * 0.5
        };
    }

    static double[] combine(double[] a, double[] b) {
        double[] out = new double[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    // MODEL
    static class NeuralNet {

        double[][] W1;
        double[] b1 = new double[HIDDEN_SIZE];

        double[][] W2h = new double[HIDDEN_SIZE][HIDDEN_SIZE];
        double[] b2h = new double[HIDDEN_SIZE];

        double[] W3 = new double[HIDDEN_SIZE];
        double b3 = 0;

        NeuralNet(int inputSize) {
            W1 = new double[inputSize][HIDDEN_SIZE];

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    W1[i][j] = rand.nextGaussian() * 0.1;

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    W2h[i][j] = rand.nextGaussian() * 0.1;

            for (int j = 0; j < HIDDEN_SIZE; j++)
                W3[j] = rand.nextGaussian() * 0.1;
        }

        double forward(double[] x) {
            double[] h1 = new double[HIDDEN_SIZE];
            double[] h2 = new double[HIDDEN_SIZE];

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = b1[j];
                for (int i = 0; i < x.length; i++)
                    sum += x[i] * W1[i][j];
                h1[j] = Math.tanh(sum);
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = b2h[j];
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    sum += h1[i] * W2h[i][j];
                h2[j] = Math.tanh(sum);
            }

            double out = b3;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                out += h2[j] * W3[j];

            double pred = 1.0 / (1.0 + Math.exp(-out));
            return Math.max(0.05, Math.min(0.95, pred));
        }

        void train(double[] x, double y) {

            double[] h1 = new double[HIDDEN_SIZE];
            double[] h2 = new double[HIDDEN_SIZE];

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = b1[j];
                for (int i = 0; i < x.length; i++)
                    sum += x[i] * W1[i][j];
                h1[j] = Math.tanh(sum);
            }

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double sum = b2h[j];
                for (int i = 0; i < HIDDEN_SIZE; i++)
                    sum += h1[i] * W2h[i][j];
                h2[j] = Math.tanh(sum);
            }

            double out = b3;
            for (int j = 0; j < HIDDEN_SIZE; j++)
                out += h2[j] * W3[j];

            double pred = 1.0 / (1.0 + Math.exp(-out));
            double gradOut = (pred - y) * pred * (1 - pred);

            for (int j = 0; j < HIDDEN_SIZE; j++)
                W3[j] -= LEARNING_RATE * gradOut * h2[j];

            b3 -= LEARNING_RATE * gradOut;

            double[] gradH2 = new double[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++)
                gradH2[j] = gradOut * W3[j] * (1 - h2[j] * h2[j]);

            for (int i = 0; i < HIDDEN_SIZE; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    W2h[i][j] -= LEARNING_RATE * gradH2[j] * h1[i];

            for (int j = 0; j < HIDDEN_SIZE; j++)
                b2h[j] -= LEARNING_RATE * gradH2[j];

            double[] gradH1 = new double[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    gradH1[i] += gradH2[j] * W2h[i][j];
                gradH1[i] *= (1 - h1[i] * h1[i]);
            }

            for (int i = 0; i < x.length; i++)
                for (int j = 0; j < HIDDEN_SIZE; j++)
                    W1[i][j] -= LEARNING_RATE * gradH1[j] * x[i];

            for (int j = 0; j < HIDDEN_SIZE; j++)
                b1[j] -= LEARNING_RATE * gradH1[j];
        }
    }

    static class Data {
        String text;
        double label;

        Data(String t, double l) {
            text = t;
            label = Math.min(1.0, l / 80.0);
        }
    }

    static List<Data> loadCSV(String path) throws Exception {
        List<Data> data = new ArrayList<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        br.readLine();

        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)");

            if (parts.length < 2) continue;

            double label;
            try {
                label = Double.parseDouble(parts[parts.length - 1].trim());
            } catch (Exception e) {
                continue;
            }

            StringBuilder text = new StringBuilder();
            for (int i = 0; i < parts.length - 1; i++)
                text.append(parts[i]).append(" ");

            data.add(new Data(text.toString(), label));
        }

        br.close();
        return data;
    }

    public static void main(String[] args) throws Exception {

        List<Data> dataset = loadCSV("resume_dataset.csv");

        buildVocab(dataset);

        int inputSize = vocab.size() + 9;
        NeuralNet model = new NeuralNet(inputSize);

        double bestMAE = Double.MAX_VALUE;

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {

            Collections.shuffle(dataset);

            double totalError = 0;
            int correct = 0;

            for (Data d : dataset) {

                double[] vec = combine(tfidfVector(d.text), extraFeatures(d.text));

                double pred = model.forward(vec);
                model.train(vec, d.label);

                double error = Math.abs(pred - d.label);
                totalError += error;

                if (Math.abs(pred - d.label) < 0.08) correct++;
            }

            double mae = totalError / dataset.size();

            if (mae < bestMAE) bestMAE = mae;

            if (epoch > 30 && mae > bestMAE + 0.005) {
                System.out.println("Early stopping at epoch " + epoch);
                break;
            }

            double accuracy = (correct * 100.0) / dataset.size();

            System.out.println("Epoch " + epoch +
                    " | MAE: " + mae +
                    " | Accuracy: " + accuracy + "%");
        }

        String testResume = "Java developer with spring boot and 2 years experience";

        double[] testVec = combine(tfidfVector(testResume), extraFeatures(testResume));
        double score = model.forward(testVec) * 100;

        System.out.println("\nFinal Resume Score: " + score);
    }
}
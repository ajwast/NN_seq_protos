import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        :param X: 2D array of shape (n_samples, n_features)
        :param y: 1D array of shape (n_samples,), binary labels (0 or 1)
        """
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.means = {c: np.mean(X[y == c], axis=0) for c in self.classes}
        self.variances = {c: np.var(X[y == c], axis=0) for c in self.classes}

    def _gaussian_pdf(self, x, mean, var):
        """
        Calculate the probability density function of a Gaussian distribution.
        """
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))

    def predict(self, X):
        """
        Predict the class for each sample in X.
        :param X: 2D array of shape (n_samples, n_features)
        :return: 1D array of predicted labels
        """
        predictions = []
        for sample in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(
                    np.log(self._gaussian_pdf(sample, self.means[c], self.variances[c]))
                )
                posteriors[c] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Example usage:
# Generate synthetic data for demonstration
np.random.seed(40)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = np.random.choice([0, 1], size=n_samples)


# Split into training and testing sets
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train and evaluate the model
model = NaiveBayesClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")


def generateSeq(num_steps):

    # Generate new data for prediction
    new_data = np.random.randn(num_steps, 5)
    seq = []
    # Make predictions for each new sample
    for i in range(num_steps):
        step = model.predict(new_data[i].reshape(1, -1))  # Reshape to 2D array
        seq.append(step[0])
    return seq


print(generateSeq(16))

print(generateSeq(16))

print(generateSeq(16))

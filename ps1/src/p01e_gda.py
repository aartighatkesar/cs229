import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to pred_path
    #x_max = np.max(x_train, axis=0)
    #x_min = np.min(x_train, axis=0)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    gda= GDA()
    #x_scaled=np.divide(x_train - x_min, x_max - x_min)
    gda.fit(x_train, y_train)
    #x_eval_scaled=np.divide(x_eval - x_min, x_max - x_min)
    output=gda.predict(x_eval)
    np.savetxt(pred_path,output)
    theta_to_plot = np.vstack((gda.theta_0, gda.theta))
    util.plot(x_train, y_train, theta_to_plot, pred_path.replace(".txt","_train.png"))
    util.plot(x_eval, y_eval, theta_to_plot, pred_path.replace(".txt", "_eval.png"))
    # *** END CODE HERE ***


class GDA(LinearModel):
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        y_zeros = 1 - y
        y_ones = y
        y_zeros = y_zeros.reshape(y_zeros.shape[0], 1)
        y_ones = y_ones.reshape(y_ones.shape[0], 1)
        phi = y_ones.sum() / y_ones.shape[0]
        mu_0 = np.dot(y_zeros.T, x) / y_zeros.sum()
        mu_1 = np.dot(y_ones.T, x) / y_ones.sum()
        x_new_for_calc = x- (y_zeros * mu_0 + y_ones * mu_1)
        sigma = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i <= j:
                    tempVal = (x_new_for_calc[:, i].reshape(x.shape[0], 1) * x_new_for_calc[:, j].reshape(x.shape[0], 1))
                    sigma[i][j] = np.sum(tempVal) / x.shape[0]
                    sigma[j][i] = sigma[i][j]
        sigma_inverse=np.linalg.inv(sigma)
        mu_0=mu_0.T
        mu_1=mu_1.T
        self.theta = np.dot(sigma_inverse, (mu_1 - mu_0))
        temp1 = np.dot(mu_0.T, sigma_inverse)
        temp1 = np.sum(np.dot(temp1, mu_0))
        temp2 = np.dot(mu_1.T, sigma_inverse)
        temp2 = np.sum(np.dot(temp2, mu_1))
        self.theta_0 = 1 / 2 * ((temp1 - temp2) - np.log((1 - phi) / phi))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        temp=util.sigmoid(np.dot(x,self.theta)+self.theta_0)
        return temp
        # *** END CODE HERE

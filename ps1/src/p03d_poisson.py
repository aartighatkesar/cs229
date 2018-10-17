import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    poisonreg = PoissonRegression(step_size=lr)
    poisonreg.fit(x_train, y_train)
    print(poisonreg.theta)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    output = poisonreg.predict(x_eval)
    np.savetxt(pred_path, output)
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        mini_batch_size=100
        if self.theta is None:
            self.theta=np.zeros(x.shape[1])
        error=1e9
        numIters=0
        while error > self.eps and numIters < self.max_iter:
            for i in range(0,x.shape[0],mini_batch_size):
                end_index=min(i+mini_batch_size,x.shape[0])
                if end_index>x.shape[0]:
                    end_index=x.shape[0]
                x_batch=x[i:end_index,:]
                theta_x=np.dot(x_batch,self.theta)
                exp_theta_x=np.exp(theta_x)
                diff=y[i:end_index]-exp_theta_x
                prod=np.dot(x_batch.T,diff)
                avg=prod/x_batch.shape[0]
                theta_new=self.theta+self.step_size*avg
                error = np.sum(np.abs(self.theta - theta_new))
                if error < self.eps:
                    break
                self.theta=theta_new.copy()
            numIters+=1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        theta_x=np.dot(x,self.theta)
        return np.exp(theta_x)
        # *** END CODE HERE ***

import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to pred_path
    x_eval, y_eval= util.load_dataset(eval_path, add_intercept=True)
    logreg=LogisticRegression()
    logreg.fit(x_train,y_train)
    output=logreg.predict(x_eval)
    np.savetxt(pred_path,output)
    #util.plot(x_train, y_train, logreg.theta, pred_path.replace(".txt","_train.png"))
    #util.plot(x_eval, y_eval, logreg.theta, pred_path.replace(".txt", "_eval.png"))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver."""


    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        if self.theta is None:
            self.theta=np.zeros((x.shape[1],1))
        y=y.reshape((y.shape[0],1))
        error=1e9
        numIters=0
        while error>self.eps and numIters<self.max_iter:
            hess=util.hessian(x,self.theta)
            Jprime=util.gradient(x,self.theta,y)
            hessInv=np.linalg.inv(hess)
            theta_new=self.theta-hessInv.dot(Jprime)
            error=np.sum(np.abs(self.theta-theta_new))
            self.theta=theta_new.copy()
            numIters+=1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return util.sigmoid(x.dot(self.theta))
        # *** END CODE HERE ***

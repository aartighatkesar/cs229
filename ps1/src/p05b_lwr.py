import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    locallyweightedlinearregression=LocallyWeightedLinearRegression(tau=tau)
    locallyweightedlinearregression.fit(x_train,y_train)
    # Get MSE value on the validation set
    x_eval, y_eval= util.load_dataset(eval_path, add_intercept=True)
    output=locallyweightedlinearregression.predict(x_eval)
    mse=np.sum((y_eval-output)**2)/y_eval.shape[0]
    #print(mse)
    #util.plot_prob5(x_train,y_train,x_eval,output,'output/p05b_plot.png')
    # Plot validation predictions on top of training set
    # No need to save anything
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        output=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            x_to_use=x[i,:]
            diff=self.x-x_to_use
            diff_squared=diff**2
            val_to_exp=np.sum(diff_squared,axis=1)
            val_to_exp=val_to_exp/(-2*self.tau*self.tau)
            W=np.exp(val_to_exp)
            xTW=self.x.T*W
            xTWx=np.dot(xTW,self.x)
            xTWx_inv=np.linalg.inv(xTWx)
            xTWy=np.dot(xTW,self.y)
            theta=np.dot(xTWx_inv,xTWy)
            predict=np.dot(theta.T,x_to_use)
            output[i]=predict
        return output
        # *** END CODE HERE ***

import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    lowest_mse=1e7
    lowest_tau=-1
    for tau in tau_values:
        locallyweightedlinearregression = LocallyWeightedLinearRegression(tau=tau)
        locallyweightedlinearregression.fit(x_train, y_train)
        output = locallyweightedlinearregression.predict(x_eval)
        mse = np.sum((y_eval - output) ** 2) / y_eval.shape[0]
        print(tau,mse)
        if mse<lowest_mse:
            lowest_mse=mse
            lowest_tau=tau
        #util.plot_prob5(x_train, y_train, x_eval, output, 'output/p05c_plot_'+str(tau).replace(".","_") +'_tau.png')
    print(lowest_tau,lowest_mse)

    # Run on the test set to get the MSE value
    x_test, y_test= util.load_dataset(test_path, add_intercept=True)
    locallyweightedlinearregression = LocallyWeightedLinearRegression(tau=lowest_tau)
    locallyweightedlinearregression.fit(x_train, y_train)
    output = locallyweightedlinearregression.predict(x_test)
    mse = np.sum((y_test-output) ** 2) / y_test.shape[0]
    print(mse)
    # Save test set predictions to pred_path
    np.savetxt(pred_path, output)
    # Plot data
    # *** END CODE HERE ***

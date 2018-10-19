import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    x_train, y_train = util.load_dataset(train_path,label_col='t',add_intercept=True)
    x_test, y_test = util.load_dataset(test_path,label_col='t',add_intercept=True)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    output = logreg.predict(x_test)
    np.savetxt(pred_path_c, output)
    #util.plot(x_test, y_test, logreg.theta, pred_path_c.replace(".txt", "_test.png"))
    # Part (d): Train on y-labels and test on true labels
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    output_test = logreg.predict(x_test)
    np.savetxt(pred_path_d, output_test)
    #util.plot(x_test, y_test, logreg.theta, pred_path_d.replace(".txt", "_test.png"))
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs
    output_val = logreg.predict(x_eval)
    #np.savetxt("c:\\Users\\tihor\\Downloads\\temp.txt", output_val)
    alpha = np.sum(np.dot(output_val.T, y_eval)) / np.sum(y_eval)
    output_test_new = np.divide(output_test, alpha)
    np.savetxt(pred_path_e, output_test_new)
    #util.plot(x_test, y_test, logreg.theta, pred_path_e.replace(".txt", "_test.png"),correction=alpha)
    # *** END CODER HERE

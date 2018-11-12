import sys
import pandas as pd

class GlobalVariables:
    def __init__(self):
        if sys.platform=='win32':
            self.DATA_DIR="C:\\Users\\tihor\\Documents\\kaggle\\quora\\"
            self.OUTPUT_DIR='submissions/'
            #self.DATA_DIR="d:\hk_futures\\"
            self.systemslash = "\\"
            self.train=pd.read_csv(self.DATA_DIR+'train.csv')
            self.test=pd.read_csv(self.DATA_DIR+'test.csv')
            # drop rows with nan
            self.train.dropna(inplace=True)
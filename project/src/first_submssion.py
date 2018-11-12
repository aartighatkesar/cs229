import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import GlobalParameters

vars=GlobalParameters.GlobalVariables()
p=vars.train['is_duplicate'].mean()
sub=pd.DataFrame({'test_id':vars.test['test_id'],'is_duplicate':p})
sub.to_csv(vars.OUTPUT_DIR+'first_submission.csv',index=None)

from xgboost import XGBClassifier
import numpy as np
class CustomXGBClassifier(XGBClassifier):
    @property
    def classes_(self):
        return np.array([-1, 0, 1])

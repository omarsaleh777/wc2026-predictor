
import joblib
import test_config
clf = joblib.load('test_clf.pkl')
print(clf.classes_)

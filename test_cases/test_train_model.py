import os
from sklearn.utils.validation import check_is_fitted
from starter.ml.model import load_model

def test_train_files():
    assert os.path.isfile(os.path.join('model', 'encoder_dtc.pkl'))
    assert os.path.isfile(os.path.join('model', 'lb_dtc.pkl'))
    assert os.path.isfile(os.path.join('model', 'model_dtc.pkl'))
 

def test_load_model():
    assert load_model(os.path.join('model', 'model_dtc.pkl'))


def test_model():
    model = load_model(os.path.join('model', 'model_dtc.pkl'))
    assert not check_is_fitted(model)

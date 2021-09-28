from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import requests
import pickle
import os
import pytest

# Note:
# Please make sure the api.py is running throughout the test
# otherwise the test will fail!

# Pre-Setup for test
test_digits = 6
X, y = load_boston(return_X_y=True)
model = Lasso(alpha=0)
model.fit(X, y)

# Save same model as pickle file
with open('./src/model_1.pkl', 'wb') as model_file:
    pickle.dump(obj=model, file=model_file)

# Map Correct references for the model to check in tests
prediction_ref = model.predict(X)
score_ref = model.score(X, y)
coef_ref = model.coef_
model_params = model.__dict__

del model

base_url = 'http://127.0.0.1:5000/'

############## Test Part ###############
data_X = pd.DataFrame(X)
data_X = data_X.to_json(orient='records')
data_y = pd.DataFrame(y)
data_y = data_y.to_json(orient='records')  # "records" bc pandas to_json() will mix up the index

def test_endpoint_con():
    """
    Function checks check endpoint of the API
    """
    response = requests.get(url=base_url + '/check')

    assert response.json() == 200


def test_predict_endpoint():
    """
    Function checks predict endpoint of the API
    """
    response = requests.post(url=base_url + '/predict/1', json=data_X)

    assert response.status_code == 200
    np.testing.assert_almost_equal(actual=np.array(response.json()), desired=prediction_ref,
                                   decimal=test_digits, verbose=True)


def test_score_endpoint():
    """
    Function checks score endpoint of the API
    """
    data_ = dict(X=data_X, y=data_y)

    response = requests.post(url=base_url + '/score/1', json=data_)
    assert response.status_code == 200
    assert round(response.json(), 6) == round(score_ref, 6)


def test_coef_endpoint(coef_refer=coef_ref):
    """
    Function checks model_coef endpoint of the API
    """
    response = requests.get(url=base_url + '/model_coef/1')
    assert response.status_code == 200
    np.testing.assert_almost_equal(actual=np.array(response.json()), desired=coef_refer,
                                   decimal=test_digits, verbose=True)


def test_parameters_endpoint(model_parameters=model_params):
    """
    Function checks model_params endpoint of the API
    """
    response = requests.get(url=base_url + '/model_params/1')
    params = response.json()

    assert response.status_code == 200
    for key in model_parameters.keys():
        if type(model_parameters[key]) == np.ndarray or type(model_parameters[key]) == list:
            np.testing.assert_almost_equal(actual=params[key], desired=model_parameters[key],
                                           decimal=test_digits, verbose=True)
        else:
            assert params[key] == model_parameters[key]

def test_update_endpoint(coef_refer=coef_ref):
    """
    Function checks update_model endpoint of the API
    """
    new_coef = np.random.uniform(size=len(coef_refer)).tolist()
    response = requests.put(url=base_url + '/update_model/1', json={'params': new_coef})
    assert response.status_code == 200

    with open('./src/model_1.pkl', 'rb') as pkl_file:
        reference_model = pickle.load(pkl_file)
        coef_refer = reference_model.coef_

    np.testing.assert_array_almost_equal(x=coef_refer, y=new_coef,
                                         decimal=test_digits, verbose=True)

def test_create_model_endpoint():
    """
    Function checks create_model endpoint of the API
    """
    models_before = os.listdir('src')

    new_params = {'alpha': 1}
    response = requests.put(url=base_url + '/create_model', json=new_params)
    new_model_id = response.json()['model_id']

    assert response.status_code == 200
    assert f'model_{new_model_id}.pkl' not in models_before
    assert f'model_{new_model_id}.pkl' in os.listdir('src')


def test_delete_endpoint():
    """
    Function checks delete_model endpoint of the API
    """
    response = requests.delete(url=base_url + '/delete_model/1')

    assert response.status_code == 200
    assert 'model.pkl' not in os.listdir()


if __name__ == '__main__':
    pytest.main()

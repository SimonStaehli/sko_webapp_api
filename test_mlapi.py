from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from api import app
import numpy as np
import pandas as pd
import pickle
import os
import pytest

app.config['TESTING'] = True
app = app.test_client()

# Note:
# Please make sure the api.py is running throughout the test
# otherwise the test will fail!

# Save same model as pickle file
# Pre-Setup for test
test_digits = 6
X, y = load_boston(return_X_y=True)

if 'model_1.pkl' not in os.listdir('src'):
    model = Ridge(alpha=0)
    model.fit(X, y)
    with open('./src/model_1.pkl', 'wb') as model_file:
        pickle.dump(obj=model, file=model_file)
else:
    with open('./src/model_1.pkl', 'rb') as model_file:
        model = pickle.load(file=model_file)

# Map Correct references for the model to check in tests
prediction_ref = model.predict(X)
score_ref = model.score(X, y)
coef_ref = model.coef_
model_params = model.__dict__

del model

############## Test Part ###############
data_X = pd.DataFrame(X)
data_X = data_X.to_json(orient='records')
data_y = pd.DataFrame(y)
data_y = data_y.to_json(orient='records')  # "records" bc pandas to_json() will mix up the index

def test_endpoint_con():
    """
    Function checks check endpoint of the API
    """
    response = app.get('/check')

    assert response.get_json() == 200


def test_predict_endpoint():
    """
    Function checks predict endpoint of the API
    """
    response = app.post('/predict/1', json=data_X)

    assert response.status_code == 200
    np.testing.assert_almost_equal(actual=np.array(response.get_json()), desired=prediction_ref,
                                   decimal=test_digits, verbose=True)


def test_score_endpoint():
    """
    Function checks score endpoint of the API
    """
    data_ = dict(X=data_X, y=data_y)

    response = app.post('/score/1', json=data_)
    assert response.status_code == 200
    assert round(response.get_json(), 6) == round(score_ref, 6)


def test_coef_endpoint(coef_refer=coef_ref):
    """
    Function checks model_coef endpoint of the API
    """
    response = app.get('/model_coef/1')
    assert response.status_code == 200
    np.testing.assert_almost_equal(actual=np.array(response.get_json()), desired=coef_refer,
                                   decimal=test_digits, verbose=True)


def test_parameters_endpoint(model_parameters=model_params):
    """
    Function checks model_params endpoint of the API
    """
    response = app.get('/model_params/1')
    params = response.get_json()

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
    response = app.put('/update_model/1', json={'params': new_coef})
    assert response.status_code == 200

    response = app.get('/model_coef/1')

    np.testing.assert_array_almost_equal(x=np.array(response.get_json()), y=new_coef,
                                         decimal=test_digits, verbose=True)

def test_create_model_endpoint():
    """
    Function checks create_model endpoint of the API
    """
    models_before = app.get('/get_all_models')

    new_params = {'alpha': 1}
    response = app.put('/create_model', json=new_params)
    new_model_id = response.get_json()['model_id']
    response = app.delete(f'/delete_model/{new_model_id}')

    assert response.status_code == 200
    assert f'model_{new_model_id}.pkl' not in models_before.get_json()


def test_delete_endpoint():
    """
    Function checks delete_model endpoint of the API
    """
    new_params = {'alpha': 1}
    response = app.put('/create_model', json=new_params)
    new_model_id = response.get_json()['model_id']

    response = app.delete(f'/delete_model/{new_model_id}')

    assert response.status_code == 200
    assert f'model_{new_model_id}.pkl' not in os.listdir()


if __name__ == '__main__':
    pytest.main()

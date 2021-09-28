from transform import CustomTransformer
from model import CustomModel
from flask import Flask
from flask import request, jsonify
import pandas as pd

app = Flask(__name__)
transformer = CustomTransformer()

@app.route('/check', methods=['GET'])
def check():
    """
    Prediction Endpoint. Returns Predictions for POST Request. Reachable by POST-Request.
    """
    if request.method == 'GET':
        return jsonify(200)

# Finish and write testscript
@app.route('/create_model', methods=['PUT'])
def create_model():
    """
    Creates a new model based on existing model with new parameters and returns ID of the new created model.
    """
    model = CustomModel(filepath='./src/model_1.pkl')

    if request.method == 'PUT':
        try:
            # Extract data from api
            new_params = request.get_json()
            print('---- Received Data Object as JSON ----')
            model_id = model.create_new_model(model_params=new_params)
        except ValueError:
            return jsonify("Please send valid parameters to create a new model.")

        return jsonify(dict(model_id=model_id))

@app.route('/predict/<id>', methods=['POST'])
def predict(id):
    """
    Prediction Endpoint. Returns Predictions for POST Request. Reachable by POST-Request.

    Parameters
    ----------
    id:
        Id of the model on server.

    Returns
    -------
    predictions:
        Predictions by the chosen model.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'POST':
        try:
            # Extract data from api
            data = request.get_json()
            data = pd.read_json(data)
            print('---- Received Data Object as JSON ----')
        except ValueError:
            return jsonify("Please enter a number.")
        # Data Transformation
        try:
            data = transformer.fit_transform(X=data)
            print('---- Transformed Data Successfully ----')
        except NotImplementedError:
            return jsonify('Something went wrong with the Transformation.')
        # Predictions
        try:
            predictions = model.model_predict(X=data)
            print('---- Data fed to the model and predictions returned ----')
        except NotImplementedError:
            return jsonify('Something went wrong with Predictions.')

        return jsonify(predictions.tolist())

@app.route('/score/<id>', methods=['POST'])
def get_model_score(id):
    """
    Scoring Endpoint. Returns Score of the model for given X and y. Reachable by POST-Request.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'POST':
        try:
            data = request.get_json()
            X = pd.read_json(data['X'], orient='records')
            X = X.to_numpy()
            y = pd.read_json(data['y'], orient='records')
            y = y.to_numpy()
            print('---- Data Collected ----')
        except ValueError:
            return jsonify('No Valid Input for Model.')
        try:
            model_score = model.model_score(X=X, y=y)
            print('---- Data fed to model and Score returned ----')
        except TypeError:
            return jsonify('Datatype not valid. Be Sure to input dict with format: {X, y}')

        return jsonify(model_score)

@app.route('/update_model/<id>', methods=['PUT'])
def update_model(id):
    """
    Update Model Endpoint. Updates Model parameters. Reachable by PUT-Request.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'PUT':
        try:
            data = request.get_json()
            model_params = data['params']
            print('---- Data Collected ----')
        except ValueError:
            return jsonify('No Valid Input for Model.')
        try:
            model.update_model_params(new_params=model_params)
            print('---- Model updated with new parameters. ----')
        except TypeError:
            return jsonify('Parameters not accepted.')

        return jsonify('Parameters updated successfully.')

@app.route('/delete_model/<id>', methods=['DELETE'])
def delete_model(id):
    """
    Delete Enpoint. Deletes the model. Reachable by DELETE-Request.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'DELETE':
        try:
            model.delete_model()
        except ValueError:
            return jsonify('No Valid Input for Model.')

        return jsonify('Model Deleted Successfully.')

@app.route('/model_coef/<id>', methods=['GET'])
def model_coef(id):
    """
    Update model parameters endpoint. Updates model parameters. Reachable by GET-Request.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'GET':
        try:
            model_coef = model.return_coef()
        except TypeError:
            return jsonify('Datatype not valid. Be Sure to input list in format: [X, y]')

        return jsonify(model_coef.tolist())

@app.route('/model_params/<id>', methods=['GET'])
def model_params(id):
    """
    Update model parameters endpoint. Updates model parameters. Reachable by GET-Request.
    """
    model = CustomModel(filepath=f'./src/model_{id}.pkl')

    if request.method == 'GET':
        try:
            model_params = model.return_parameters()
        except TypeError:
            return jsonify('Datatype not valid. Be Sure to input list in format: [X, y]')

        return jsonify(model_params)



if __name__ == '__main__':
    app.run(debug=True)
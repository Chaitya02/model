from flask import Flask, request, jsonify
import pickle
import numpy as np
# from sklearn.preprocessing import StandardScaler
import pandas as pd

categories = ["name", "screen_name", "statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count", "geo_enabled", "profile_use_background_image", "profile_background_tile"]

data = pd.read_csv('dataset.csv')
# scaler = StandardScaler()
# scaler.fit(data[categories])
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)


@app.route("/", methods=['POST'])
def hello_world():
    input_data = request.json
    input = np.array([input_data[x] for x in categories])
    input = input.reshape(1, -1)
    # input = scaler.transform(input)
    print(input)
    prediction = model.predict([[input]])
    print(prediction, type(prediction))

    return jsonify({"prediction": prediction.tolist() })

if __name__ == "__main__":
    app.run(debug=True)
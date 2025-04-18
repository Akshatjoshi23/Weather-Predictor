from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

# Load ML model
model = pickle.load(open("weather_model.pkl", "rb"))

API_KEY = "6e6f9659fef62e5c5d1103979100d281"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    city = None
    actual_temp = None
    actual_humidity = None
    wind_speed = None
    hourly_forecast = []
    tomorrow_forecast = None

    if request.method == 'POST':
        city = request.form['city'].strip()

        # --- Current Weather ---
        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        current_response = requests.get(current_url)
        current_data = current_response.json()
        print("Current:", current_data)

        if current_data.get('cod') != 200:
            prediction = "City not found."
        else:
            actual_temp = current_data['main']['temp']
            actual_humidity = current_data['main']['humidity']
            wind_speed = current_data['wind']['speed']
            windy = 1 if wind_speed > 4 else 0

            features = np.array([[actual_temp, actual_humidity, windy]])
            predicted_weather = model.predict(features)[0]
            prediction = f"Predicted Weather: {predicted_weather}"

            # --- Forecast Data ---
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
            forecast_response = requests.get(forecast_url)
            forecast_data = forecast_response.json()
            print("Forecast:", forecast_data)

            if forecast_data.get('cod') == '200':
                now = datetime.utcnow()
                tomorrow = now + timedelta(days=1)
                tomorrow_date = tomorrow.strftime('%Y-%m-%d')

                for entry in forecast_data['list']:
                    dt_txt = entry['dt_txt']
                    time_obj = datetime.strptime(dt_txt, '%Y-%m-%d %H:%M:%S')

                    # Next 24 hrs forecast (3-hour intervals)
                    if now <= time_obj <= now + timedelta(hours=24):
                        hourly_forecast.append({
                            'time': time_obj.strftime('%I:%M %p'),
                            'temp': entry['main']['temp'],
                            'condition': entry['weather'][0]['description']
                        })

                    # Tomorrow (noon forecast)
                    if dt_txt.startswith(tomorrow_date) and '12:00:00' in dt_txt:
                        tomorrow_forecast = {
                            'temp': entry['main']['temp'],
                            'condition': entry['weather'][0]['description']
                        }

    return render_template(
        'index.html',
        prediction=prediction,
        city=city,
        temp=actual_temp,
        humidity=actual_humidity,
        wind=wind_speed,
        hourly_forecast=hourly_forecast,
        tomorrow_forecast=tomorrow_forecast
    )

if __name__ == '__main__':
    app.run(debug=True)

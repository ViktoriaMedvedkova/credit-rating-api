import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Это важно для разрешения запросов с вашего телефона

# --- Загружаем модель ---
model = joblib.load('credit_rating_model.pkl')
print("✅ Модель загружена!")

# --- Эндпоинт для проверки работы сервера ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'ML service is running'})

# --- Основной эндпоинт для расчёта рейтинга ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"📥 Данные получены: {data}")

        # Извлечение признаков
        features = pd.DataFrame([[
            data['age'],
            data['workExperience'],
            data['income'],
            data['activeLoans'],
            data['latePayments'],
            data['monthlyPayment']
        ]], columns=['age', 'work_experience', 'income', 'active_loans', 'late_payments', 'monthly_payment'])

        # Предсказание
        rating = model.predict(features)[0]
        rating = int(rating)
        print(f"📤 Предсказан рейтинг: {rating}")

        return jsonify({'credit_rating': rating, 'message': f'Ваш кредитный рейтинг: {rating}'})
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from joblib import load

# presence_classifier = load('presence_classifier.joblib')
# presence_vect = load('presence_vectorizer.joblib')
# category_classifier = load('category_classifier.joblib')
# category_vect = load('category_vectorizer.joblib')

# app = Flask(__name__)
# CORS(app)

# @app.route('/', methods=['GET'])
# def index():
#     return "Flask server is running!"
# @app.route('/test_post', methods=['POST'])
# def test_post():
#     data = request.get_json()  # Parse JSON data from the request body
#     return jsonify({"message": "POST request successful", "data_received": data}), 200

# @app.route('/', methods=['POST'])
# def main():
#     if request.method == 'POST':
#         output = []
#         data = request.get_json().get('tokens')

#         for token in data:
#             result = presence_classifier.predict(presence_vect.transform([token]))
#             if result == 'Dark':
#                 cat = category_classifier.predict(category_vect.transform([token]))
#                 output.append(cat[0])
#             else:
#                 output.append(result[0])

#         dark = [data[i] for i in range(len(output)) if output[i] == 'Dark']
#         for d in dark:
#             print(d)
#         print()
#         print(len(dark))

#         message = '{ \'result\': ' + str(output) + ' }'
#         print(message)

#         json = jsonify(message)

#         return json

# if __name__ == '__main__':
#     app.run(threaded=True, debug=True)


# from flask import Flask, jsonify, request
# from flask_cors import CORS
# from joblib import load
# import traceback

# app = Flask(__name__)
# CORS(app)

# # Load classifiers
# try:
#     presence_classifier = load('presence_classifier.joblib')
#     presence_vect = load('presence_vectorizer.joblib')
#     category_classifier = load('category_classifier.joblib')
#     category_vect = load('category_vectorizer.joblib')
# except Exception as e:
#     print(f"Error loading classifiers: {str(e)}")
#     raise

# @app.route('/', methods=['POST'])
# def main():
#     try:
#         if request.method == 'POST':
#             output = []
#             data = request.get_json()
            
#             if not data or 'tokens' not in data:
#                 return jsonify({'error': 'No tokens provided'}), 400
                
#             tokens = data.get('tokens', [])

#             for token in tokens:
#                 result = presence_classifier.predict(presence_vect.transform([token]))
#                 if result[0] == 'Dark':
#                     cat = category_classifier.predict(category_vect.transform([token]))
#                     output.append(cat[0])
#                 else:
#                     output.append(result[0])

#             dark = [tokens[i] for i in range(len(output)) if output[i] != 'Not Dark']
#             print(f"Found {len(dark)} dark patterns")
#             for d in dark:
#                 print(d)

#             return jsonify({'result': output})
            
#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         print(traceback.format_exc())
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5000, threaded=True, debug=True)
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from joblib import load
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load classifiers with error handling
try:
    presence_classifier = load('presence_classifier.joblib')
    presence_vect = load('presence_vectorizer.joblib')
    category_classifier = load('category_classifier.joblib')
    category_vect = load('category_vectorizer.joblib')
except Exception as e:
    logger.error(f"Error loading classifiers: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/', methods=['POST'])
def main():
    try:
        data = request.get_json()
        
        if not data or 'tokens' not in data:
            return jsonify({'error': 'No tokens provided'}), 400
            
        tokens = data.get('tokens', [])
        output = []

        for token in tokens:
            result = presence_classifier.predict(presence_vect.transform([token]))
            if result[0] == 'Dark':
                cat = category_classifier.predict(category_vect.transform([token]))
                output.append(cat[0])
            else:
                output.append(result[0])

        dark = [tokens[i] for i in range(len(output)) if output[i] != 'Not Dark']
        logger.info(f"Found {len(dark)} dark patterns")
        for d in dark:
            logger.info(d)

        return jsonify({'result': output})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # For production, use gunicorn instead
    app.run(
        host='0.0.0.0', 
        port=int(os.environ.get('PORT', 5000)), 
        threaded=True
    )
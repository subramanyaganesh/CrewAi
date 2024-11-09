from flask import Flask, request, jsonify
from AzureWorkFlowSecond import process_query  # This would be your Azure workflow file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/process', methods=['POST'])
def process_request():
    # print("|||||||||||||||||||||||||||||||||||||||||||||||")
    # print('this is the entryyyyy')
    # print("|||||||||||||||||||||||||||||||||||||||||||||||")
    try:
        # Get the JSON data from the request body
        print("|||||||||||||||||||||||||||||||||||||||||||||||")
        print('this is the request',dir(request))
        print("|||||||||||||||||||||||||||||||||||||||||||||||")

        data = request.get_json()
        
        # Check if query exists in the request
        if not data or 'query' not in data:
            return jsonify({
                'error': 'No query provided',
                'status': 'error'
            }), 400
        
        # Extract the query
        query = data['query']
        
        
        # Process the query through Azure workflow
        try:
            result = process_query(query)

            print("this is the result",result)
            
            # Return the result
            return jsonify({
                'answer': result,
                'status': 'success'
            }), 200
            
        except Exception as e:
            return jsonify({
                'error': f'Azure workflow processing error: {str(e)}',
                'status': 'error'
            }), 500
            
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
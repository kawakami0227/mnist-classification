from flask import Flask
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app)

@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

@api.route('/pre')
class prediction(Resource):
    def get(self):
        return {'prediction': '2'}

if __name__ == '__main__':
    app.run(debug=True)
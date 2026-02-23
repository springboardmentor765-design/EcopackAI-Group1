from flask import Flask
from flask_cors import CORS
from routes.product_routes import product_bp

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Register blueprint
app.register_blueprint(product_bp, url_prefix="/product")

if __name__ == "__main__":
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(rule)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )

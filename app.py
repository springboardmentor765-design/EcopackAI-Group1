from flask import Flask
from routes.recommendation_routes import recommend_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Register blueprint
app.register_blueprint(recommend_bp, url_prefix="/recommend")

# Health check route (very useful for Render testing)
@app.route("/")
def home():
    return {"message": "EcoPack Backend is running successfully!"}

if __name__ == "__main__":
    # For local development only
    app.run(host="0.0.0.0", port=5000)
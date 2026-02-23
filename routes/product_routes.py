from flask import Blueprint, request, jsonify
from database.db import get_connection

# Create Blueprint
product_bp = Blueprint("product_bp", __name__)

# API to add product
@product_bp.route("/api/product", methods=["POST"])
def add_product():
    try:
        data = request.get_json()  # Get JSON data from request

        # Connect to DB
        conn = get_connection()
        cur = conn.cursor()

        # Insert into products table
        cur.execute("""
            INSERT INTO products (category, weight, shipping_category, fragile)
            VALUES (%s, %s, %s, %s)
            RETURNING product_id
        """, (
            data["category"],
            data["weight"],
            data["shipping_category"],
            data["fragile"]
        ))

        product_id = cur.fetchone()[0]
        conn.commit()

        cur.close()
        conn.close()

        return jsonify({"status": "success", "product_id": product_id})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

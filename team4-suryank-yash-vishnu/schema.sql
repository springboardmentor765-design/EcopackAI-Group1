-- EcoPackAI Database Schema

-- Drop existing tables
DROP TABLE IF EXISTS material_suitability CASCADE;
DROP TABLE IF EXISTS materials CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- Materials table
CREATE TABLE materials (
    id SERIAL PRIMARY KEY,
    material_type VARCHAR(255) NOT NULL,
    tensile_strength_mpa FLOAT,
    weight_capacity_kg FLOAT,
    biodegradability_score FLOAT,
    co2_emission_score FLOAT,
    recyclability_percent FLOAT,
    moisture_barrier_grade INT,
    ai_recommendation VARCHAR(255),
    co2_impact_index FLOAT,
    cost_efficiency_index FLOAT,
    material_suitability_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_category VARCHAR(255) NOT NULL,
    packaging_requirement VARCHAR(500),
    typical_weight_kg FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Material-Product mapping table
CREATE TABLE material_suitability (
    material_id INT REFERENCES materials(id) ON DELETE CASCADE,
    product_id INT REFERENCES products(id) ON DELETE CASCADE,
    suitability_override FLOAT,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (material_id, product_id)
);

-- ============================================================================
-- EcoPackAI - Complete PostgreSQL Database Setup
-- ============================================================================
-- Execute these commands in order to create the complete database structure
-- ============================================================================

-- STEP 1: Create the database (run this first in psql or pgAdmin)
-- ============================================================================
-- Note: You need to be connected to the default 'postgres' database to create a new database
-- In psql: psql -U postgres
-- Then run:

CREATE DATABASE ecopack_db;

-- After creating the database, connect to it:
-- In psql: \c ecopack_db
-- In pgAdmin: Right-click on ecopack_db and select "Query Tool"

-- ============================================================================
-- STEP 2: Drop existing tables (if any) - Clean slate
-- ============================================================================

DROP TABLE IF EXISTS material_suitability CASCADE;
DROP TABLE IF EXISTS materials CASCADE;
DROP TABLE IF EXISTS products CASCADE;

-- ============================================================================
-- STEP 3: Create the Materials table (Core table)
-- ============================================================================

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

-- Add comments to document the table
COMMENT ON TABLE materials IS 'Stores eco-friendly packaging material properties and sustainability scores';
COMMENT ON COLUMN materials.material_type IS 'Type of material (e.g., Bioplastic, Recycled Paper)';
COMMENT ON COLUMN materials.tensile_strength_mpa IS 'Tensile strength in megapascals';
COMMENT ON COLUMN materials.weight_capacity_kg IS 'Maximum weight capacity in kilograms';
COMMENT ON COLUMN materials.biodegradability_score IS 'Biodegradability score (0-100)';
COMMENT ON COLUMN materials.co2_emission_score IS 'CO2 emission score (lower is better)';
COMMENT ON COLUMN materials.recyclability_percent IS 'Recyclability percentage (0-100)';
COMMENT ON COLUMN materials.co2_impact_index IS 'Calculated CO2 impact index (higher is better)';
COMMENT ON COLUMN materials.cost_efficiency_index IS 'Calculated cost efficiency index';
COMMENT ON COLUMN materials.material_suitability_score IS 'Overall suitability score for ranking';

-- ============================================================================
-- STEP 4: Create the Products table (For future product categorization)
-- ============================================================================

CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    product_category VARCHAR(255) NOT NULL,
    packaging_requirement VARCHAR(500),
    typical_weight_kg FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE products IS 'Stores product categories and their packaging requirements';
COMMENT ON COLUMN products.product_category IS 'Product category (e.g., Electronics, Food, Cosmetics)';
COMMENT ON COLUMN products.packaging_requirement IS 'Specific packaging needs (e.g., Moisture barrier, Shock absorption)';

-- ============================================================================
-- STEP 5: Create the Material Suitability junction table
-- ============================================================================

CREATE TABLE material_suitability (
    material_id INT REFERENCES materials(id) ON DELETE CASCADE,
    product_id INT REFERENCES products(id) ON DELETE CASCADE,
    suitability_override FLOAT,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (material_id, product_id)
);

COMMENT ON TABLE material_suitability IS 'Maps materials to products with custom suitability scores';
COMMENT ON COLUMN material_suitability.suitability_override IS 'Custom suitability score for this material-product pair';
COMMENT ON COLUMN material_suitability.reason IS 'Explanation for the suitability score';

-- ============================================================================
-- STEP 6: Create indexes for better query performance
-- ============================================================================

-- Index on material_type for filtering by material
CREATE INDEX idx_materials_type ON materials(material_type);

-- Index on suitability score for ranking queries
CREATE INDEX idx_materials_suitability ON materials(material_suitability_score DESC);

-- Index on CO2 emission for environmental filtering
CREATE INDEX idx_materials_co2 ON materials(co2_emission_score);

-- Index on biodegradability for sustainability queries
CREATE INDEX idx_materials_biodegradability ON materials(biodegradability_score DESC);

-- Composite index for common filter combinations
CREATE INDEX idx_materials_eco_scores ON materials(co2_emission_score, biodegradability_score, recyclability_percent);

-- ============================================================================
-- STEP 7: Insert sample product categories (Optional - for future use)
-- ============================================================================

INSERT INTO products (product_category, packaging_requirement, typical_weight_kg) VALUES
('Electronics', 'Shock absorption, Anti-static, Moisture barrier', 2.5),
('Food & Beverages', 'Food-grade, Moisture barrier, Sealable', 1.0),
('Cosmetics', 'Lightweight, Aesthetic appeal, Moisture resistant', 0.5),
('Pharmaceuticals', 'Tamper-proof, Light-blocking, Moisture barrier', 0.3),
('Clothing & Textiles', 'Breathable, Lightweight, Recyclable', 1.5),
('Industrial Parts', 'Heavy-duty, Shock absorption, Reusable', 10.0),
('Books & Documents', 'Lightweight, Recyclable, Moisture resistant', 0.8),
('Toys', 'Child-safe, Transparent, Recyclable', 0.6);

-- ============================================================================
-- STEP 8: Verify the database structure
-- ============================================================================

-- List all tables
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public';

-- Check materials table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'materials'
ORDER BY ordinal_position;

-- Check indexes
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'materials';

-- ============================================================================
-- STEP 9: Grant permissions (if using a specific user)
-- ============================================================================

-- If you created a specific user for the application, grant permissions:
-- GRANT ALL PRIVILEGES ON DATABASE ecopack_db TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- ============================================================================
-- STEP 10: Ready to load data
-- ============================================================================

-- At this point, your database structure is complete.
-- Run the Python script to load data:
-- python load_to_db.py

-- After loading, verify the data:
SELECT COUNT(*) as total_materials FROM materials;
SELECT material_type, COUNT(*) as count 
FROM materials 
GROUP BY material_type 
ORDER BY count DESC 
LIMIT 10;

-- ============================================================================
-- USEFUL QUERIES FOR YOUR PROJECT
-- ============================================================================

-- Top 10 most sustainable materials
-- SELECT material_type, material_suitability_score, co2_emission_score, biodegradability_score
-- FROM materials
-- ORDER BY material_suitability_score DESC
-- LIMIT 10;

-- Materials with low CO2 emissions
-- SELECT material_type, co2_emission_score, co2_impact_index
-- FROM materials
-- WHERE co2_emission_score < 5
-- ORDER BY co2_emission_score;

-- High-strength materials for heavy products
-- SELECT material_type, tensile_strength_mpa, weight_capacity_kg
-- FROM materials
-- WHERE weight_capacity_kg > 50
-- ORDER BY tensile_strength_mpa DESC;

-- Materials by biodegradability
-- SELECT material_type, biodegradability_score, recyclability_percent
-- FROM materials
-- WHERE biodegradability_score > 80
-- ORDER BY biodegradability_score DESC;

-- ============================================================================
-- END OF SETUP
-- ============================================================================

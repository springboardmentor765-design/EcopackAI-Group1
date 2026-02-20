SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'recommendations'
AND column_name = 'product_id';

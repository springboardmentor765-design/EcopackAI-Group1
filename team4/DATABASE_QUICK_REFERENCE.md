# EcoPackAI Database Setup - Quick Reference

## Execute in Order:

### 1. Create Database (in default postgres connection)
```sql
CREATE DATABASE ecopack_db;
```

### 2. Connect to the new database
```bash
# In psql:
\c ecopack_db

# In pgAdmin: 
# Right-click on ecopack_db → Query Tool
```

### 3. Run the complete setup
```sql
-- Copy and paste the entire contents of database_setup_complete.sql
-- OR run it directly:
```

```bash
psql -U postgres -d ecopack_db -f database_setup_complete.sql
```

### 4. Load your data
```bash
python load_to_db.py
```

### 5. Verify
```sql
SELECT COUNT(*) FROM materials;
SELECT * FROM materials LIMIT 5;
```

## Quick Command Reference

### Connect to database:
```bash
psql -U postgres -d ecopack_db
```

### List all tables:
```sql
\dt
```

### Describe a table:
```sql
\d materials
```

### Exit psql:
```sql
\q
```

### Run SQL file:
```bash
psql -U postgres -d ecopack_db -f database_setup_complete.sql
```

## Files Created:
- `database_setup_complete.sql` - Complete database setup with all tables, indexes, and sample data
- `schema.sql` - Core schema (used by load_to_db.py)
- `load_to_db.py` - Python script to load CSV data into PostgreSQL

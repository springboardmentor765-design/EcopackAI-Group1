# PostgreSQL Setup Guide for EcoPackAI

Follow these steps to connect PostgreSQL with your EcoPackAI project.

## Prerequisites

Ensure you have PostgreSQL installed on your system. If not, download and install it from [postgresql.org](https://www.postgresql.org/download/).

---

## Step 1: Install PostgreSQL (if not already installed)

### Windows:
1. Download the PostgreSQL installer from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run the installer and follow the setup wizard
3. **Remember the password** you set for the `postgres` user during installation
4. Default port is `5432` (keep this unless you have a specific reason to change it)

---

## Step 2: Create a Database for EcoPackAI

### Option A: Using pgAdmin (GUI)
1. Open **pgAdmin** (installed with PostgreSQL)
2. Connect to your PostgreSQL server (use the password you set during installation)
3. Right-click on **Databases** → **Create** → **Database**
4. Name it `ecopack_db`
5. Click **Save**

### Option B: Using Command Line (psql)
1. Open Command Prompt or PowerShell
2. Run the following command:
   ```bash
   psql -U postgres
   ```
3. Enter your PostgreSQL password when prompted
4. Create the database:
   ```sql
   CREATE DATABASE ecopack_db;
   ```
5. Exit psql:
   ```sql
   \q
   ```

---

## Step 3: Configure Database Credentials

1. In your project directory (`c:\Users\lalit\Desktop\EcoPackAI`), you'll find a file named `.env.example`
2. **Copy** `.env.example` and rename it to `.env`
3. Open `.env` in a text editor and fill in your credentials:

```env
DB_NAME=ecopack_db
DB_USER=postgres
DB_PASSWORD=your_actual_password_here
DB_HOST=localhost
DB_PORT=5432
```

**Important:** Replace `your_actual_password_here` with the password you set for the `postgres` user.

---

## Step 4: Install Required Python Libraries

The required libraries (`psycopg2-binary` and `python-dotenv`) are already installed. If you need to reinstall them:

```bash
pip install psycopg2-binary python-dotenv
```

---

## Step 5: Load Data into PostgreSQL

Once your `.env` file is configured, run the data loading script:

```bash
python load_to_db.py
```

This script will:
- Connect to your PostgreSQL database
- Create the necessary tables using `schema.sql`
- Load all 5,000 records from `materials_processed_milestone1.csv` into the `materials` table

---

## Step 6: Verify the Data

### Option A: Using pgAdmin
1. Open **pgAdmin**
2. Navigate to: **Servers** → **PostgreSQL** → **Databases** → **ecopack_db** → **Schemas** → **public** → **Tables** → **materials**
3. Right-click on **materials** → **View/Edit Data** → **All Rows**

### Option B: Using psql
1. Connect to the database:
   ```bash
   psql -U postgres -d ecopack_db
   ```
2. Run a query to check the data:
   ```sql
   SELECT COUNT(*) FROM materials;
   SELECT * FROM materials LIMIT 5;
   ```
3. Exit:
   ```sql
   \q
   ```

---

## Troubleshooting

### Issue: "psql: error: connection to server at "localhost" (::1), port 5432 failed"
- **Solution:** Ensure PostgreSQL service is running. On Windows, check Services (search for "Services" in Start menu) and look for "postgresql-x64-XX". Start it if it's stopped.

### Issue: "password authentication failed for user postgres"
- **Solution:** Double-check the password in your `.env` file matches the one you set during PostgreSQL installation.

### Issue: "database 'ecopack_db' does not exist"
- **Solution:** Create the database using Step 2 above.

### Issue: Python can't find psycopg2
- **Solution:** Run `pip install psycopg2-binary` again.

---

## Next Steps

Once your data is loaded into PostgreSQL, you can:
- Query the data using SQL
- Build a Flask/FastAPI backend to serve the data
- Create analytics dashboards
- Integrate with BI tools like Power BI or Tableau

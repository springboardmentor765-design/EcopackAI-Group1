# Deploying EcoPackAI to Render

Complete guide to host your project on Render from GitHub.

---

## Prerequisites

1. GitHub account with your EcoPackAI repository
2. Render account (free tier available at [render.com](https://render.com))
3. PostgreSQL database (Render provides free tier)

---

## Step 1: Prepare Your Repository

### 1.1 Create a Web Service File

Since you have a Flask app (`app.py`), you need to tell Render how to run it.

Create `render.yaml` in your project root:

```yaml
services:
  - type: web
    name: ecopackai
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
```

### 1.2 Update requirements.txt

Add these lines to your `requirements.txt`:

```
gunicorn
flask
```

### 1.3 Commit and Push to GitHub

```bash
git add .
git commit -m "Add Render deployment config"
git push origin main
```

---

## Step 2: Set Up PostgreSQL Database on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"PostgreSQL"**
3. Configure:
   - **Name:** `ecopackai-db`
   - **Database:** `ecopack_db`
   - **User:** (auto-generated)
   - **Region:** Choose closest to you
   - **Plan:** Free
4. Click **"Create Database"**
5. **Save these credentials** (you'll need them):
   - Internal Database URL
   - External Database URL
   - Username
   - Password

---

## Step 3: Deploy Web Service on Render

### 3.1 Create Web Service

1. Go to Render Dashboard
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository:
   - Click **"Connect GitHub"**
   - Authorize Render
   - Select your **EcoPackAI** repository

### 3.2 Configure Web Service

**Basic Settings:**
- **Name:** `ecopackai`
- **Region:** Same as database
- **Branch:** `main`
- **Root Directory:** (leave blank)
- **Runtime:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

**Instance Type:**
- Select **"Free"**

### 3.3 Add Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

Add these variables (use your database credentials from Step 2):

```
DB_NAME=ecopack_db
DB_USER=your_db_username
DB_PASSWORD=your_db_password
DB_HOST=your_db_host.render.com
DB_PORT=5432
```

**Important:** Use the **Internal Database URL** hostname for `DB_HOST`

### 3.4 Deploy

1. Click **"Create Web Service"**
2. Render will automatically:
   - Clone your repository
   - Install dependencies
   - Start your app
3. Wait 5-10 minutes for deployment

---

## Step 4: Initialize Database

Once deployed, you need to load your data into the PostgreSQL database.

### Option A: Using Render Shell (Recommended)

1. Go to your web service dashboard
2. Click **"Shell"** tab
3. Run these commands:

```bash
python load_to_db.py
```

### Option B: Using Local Connection

1. Get the **External Database URL** from your database dashboard
2. Update your local `.env` file with Render database credentials
3. Run locally:

```bash
python load_to_db.py
```

---

## Step 5: Verify Deployment

1. Go to your web service dashboard
2. Click the **URL** (e.g., `https://ecopackai.onrender.com`)
3. Your app should be live!

---

## Common Issues & Solutions

### Issue 1: Build Failed
**Solution:** Check `requirements.txt` has all dependencies

### Issue 2: App Crashes on Start
**Solution:** 
- Check logs in Render dashboard
- Verify `gunicorn` is in requirements.txt
- Ensure `app.py` has `app` variable

### Issue 3: Database Connection Failed
**Solution:**
- Use **Internal Database URL** for `DB_HOST`
- Verify environment variables are set correctly
- Check database is in same region as web service

### Issue 4: Free Tier Limitations
**Note:** 
- Free web services sleep after 15 min of inactivity
- First request after sleep takes ~30 seconds
- Free PostgreSQL has 1GB storage limit

---

## File Checklist

Before deploying, ensure you have:

- ✅ `requirements.txt` (with gunicorn, flask)
- ✅ `app.py` (your Flask application)
- ✅ `.gitignore` (excludes .env)
- ✅ `.env.example` (template for env vars)
- ✅ `schema.sql` (database schema)
- ✅ `load_to_db.py` (data loading script)

---

## Optional: Custom Domain

1. Go to web service settings
2. Click **"Custom Domain"**
3. Add your domain
4. Update DNS records as instructed

---

## Monitoring & Logs

- **Logs:** Dashboard → Your Service → "Logs" tab
- **Metrics:** Dashboard → Your Service → "Metrics" tab
- **Shell Access:** Dashboard → Your Service → "Shell" tab

---

## Cost

- **Web Service (Free):** 750 hours/month
- **PostgreSQL (Free):** 1GB storage, expires after 90 days
- **Upgrade:** $7/month for persistent database

---

Your EcoPackAI project will be live at: `https://ecopackai.onrender.com`

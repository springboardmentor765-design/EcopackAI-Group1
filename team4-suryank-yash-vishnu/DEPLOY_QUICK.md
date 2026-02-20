# Quick Render Deployment Steps

## 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Render deployment"
git push origin main
```

## 2. Create PostgreSQL Database
1. Go to [render.com](https://render.com)
2. New + → PostgreSQL
3. Name: `ecopackai-db`
4. Plan: Free
5. Create Database
6. **Save credentials**

## 3. Deploy Web Service
1. New + → Web Service
2. Connect GitHub → Select EcoPackAI repo
3. Settings:
   - Name: `ecopackai`
   - Build: `pip install -r requirements.txt`
   - Start: `gunicorn app:app`
   - Plan: Free

4. Environment Variables:
   ```
   DB_NAME=ecopack_db
   DB_USER=(from database)
   DB_PASSWORD=(from database)
   DB_HOST=(internal hostname from database)
   DB_PORT=5432
   ```

5. Create Web Service

## 4. Load Data
After deployment:
1. Go to web service → Shell
2. Run: `python load_to_db.py`

## 5. Access Your App
Your app will be live at: `https://ecopackai.onrender.com`

---

**Note:** Free tier sleeps after 15 min inactivity. First request takes ~30s to wake up.

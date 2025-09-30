# Deployment Guide for Option Pricing Dashboard

## Quick Deploy to Streamlit Community Cloud (Recommended - FREE)

### Prerequisites:
1. GitHub account
2. This repository pushed to GitHub (public or private)

### Steps:

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit dashboard"
   git push
   ```

2. **Go to Streamlit Community Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy your app:**
   - Click "New app"
   - Select your repository: `[your-username]/PDE-option-pricing`
   - Main file path: `option_pricing_app.py`
   - Click "Deploy"

4. **Wait 2-3 minutes** for deployment to complete

5. **Get your shareable link:**
   - Format: `https://[your-app-name].streamlit.app/`
   - Share this link with anyone!

### Features on Streamlit Cloud:
- ✅ Free hosting
- ✅ Automatic HTTPS
- ✅ Auto-restart on code changes
- ✅ Public or private apps
- ✅ 1GB RAM, 1 CPU core
- ✅ 1GB storage

### Important Notes:

**Performance Considerations:**
- Cloud has limited resources (1GB RAM)
- PDE calculations may be slower than local
- ML models work great (they're fast)
- Reduce grid sizes if needed (N_S=50, N_t=500)

**Model Files:**
- ML models in `data/models/` will be deployed
- If models are too large (>1GB total), consider:
  - Training smaller models
  - Using model compression
  - Hosting models separately (S3, etc.)

**Environment Variables:**
- For API keys (FRED, etc.), use Streamlit Secrets
- Go to app settings → Secrets
- Add in TOML format:
  ```toml
  FRED_API_KEY = "your-key-here"
  ```

## Alternative Deployment Options

### Option 2: Heroku (More Control)

1. **Install Heroku CLI:**
   ```bash
   brew install heroku/brew/heroku
   ```

2. **Create files:**
   - `Procfile`: `web: streamlit run option_pricing_app.py --server.port=$PORT`
   - `setup.sh`: See example below

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

**Cost:** Free tier available, paid tiers for more resources

### Option 3: Hugging Face Spaces (FREE)

1. **Go to:** https://huggingface.co/spaces
2. **Create new Space**
3. **Select:** Streamlit SDK
4. **Push your code** (they provide git instructions)

**Advantages:** Free GPU access, good for ML models

### Option 4: Railway (Simple Deployment)

1. **Go to:** https://railway.app/
2. **Connect GitHub repo**
3. **Auto-detects Streamlit**
4. **Deploy with one click**

**Cost:** $5/month after free trial

### Option 5: DigitalOcean App Platform

1. **Go to:** https://www.digitalocean.com/products/app-platform
2. **Connect repo**
3. **Configure:**
   - Buildpack: Python
   - Run command: `streamlit run option_pricing_app.py`

**Cost:** Starting at $5/month

### Option 6: AWS/GCP/Azure (Advanced)

**AWS Elastic Beanstalk:**
```bash
eb init -p python-3.9 option-pricing-app
eb create option-pricing-env
eb deploy
```

**Cost:** Varies, more control over resources

## Optimization for Cloud Deployment

### Reduce Memory Usage:

Edit `option_pricing_app.py`:
```python
# For cloud deployment, use smaller grids
pde = BlackScholesPDE(
    S_max=3*K,
    T=T,
    r=r,
    sigma=sigma,
    N_S=50,   # Reduced from 100
    N_t=500   # Reduced from 1000
)
```

### Use ML Models Primarily:

ML models are much faster and use less memory. Prioritize them for cloud deployment.

### Cache Everything:

Already implemented with `@st.cache_resource` for model loading.

## Monitoring Your Deployed App

### Streamlit Cloud:
- View logs in the app dashboard
- Monitor resource usage
- See visitor analytics

### Custom Analytics:
Add to `option_pricing_app.py`:
```python
import streamlit as st

# Track page views
if 'views' not in st.session_state:
    st.session_state.views = 0
st.session_state.views += 1
```

## Troubleshooting Deployment

**App crashes on startup:**
- Check logs for missing dependencies
- Verify all files are committed
- Check requirements.txt is complete

**Slow performance:**
- Reduce grid sizes (N_S, N_t)
- Use ML models instead of PDE
- Enable caching everywhere

**Out of memory:**
- Reduce number of test cases in Model Performance tab
- Limit historical data fetches
- Use sparse matrices for PDE solvers

**Module not found errors:**
- Add missing packages to requirements.txt
- Ensure all imports are at the top
- Check for typos in package names

## Security Considerations

**For Public Deployment:**
- Don't commit API keys
- Use Streamlit Secrets for sensitive data
- Add rate limiting if needed
- Consider authentication for private apps

**Streamlit Authentication (Community Cloud):**
- Go to app settings
- Enable "Require sign-in"
- Invite specific users

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Streamlit Cloud | Yes (unlimited) | N/A | Quick demos, personal projects |
| Heroku | Yes (limited) | $7/month | Small apps, simple deployment |
| Hugging Face | Yes (unlimited) | $9/month | ML-heavy apps, GPU access |
| Railway | $5 credit | $5+/month | Professional deployment |
| DigitalOcean | No | $5+/month | Full control, scaling |
| AWS/GCP/Azure | Limited | Varies | Enterprise, high traffic |

## Recommended: Start with Streamlit Community Cloud

It's free, simple, and perfect for sharing your option pricing dashboard!

**Support:**
- Streamlit docs: https://docs.streamlit.io/
- Community forum: https://discuss.streamlit.io/
- GitHub issues: https://github.com/streamlit/streamlit
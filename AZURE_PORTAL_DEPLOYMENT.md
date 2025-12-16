# Azure Portal Deployment (No CLI Required)

## 🌐 Deploy Using Azure Portal Web Interface

This guide shows how to deploy your Threat Severity Model to Azure using only the web interface - **no Azure CLI installation needed!**

---

## Step 1: Deploy API to Azure Container Apps via Portal

### 1.1 Create Azure Account (if needed)
- Visit: https://portal.azure.com
- Sign in with your Microsoft account
- Activate free trial ($200 credit) or use existing subscription

### 1.2 Create Container Registry (ACR)
1. Go to **Azure Portal** → Search "**Container registries**"
2. Click **+ Create**
3. Fill in:
   - **Resource Group:** Create new → `threat-model-rg`
   - **Registry name:** `threatmodelacr` (must be unique globally)
   - **Location:** East US
   - **SKU:** Basic ($5/month)
4. Click **Review + Create** → **Create**

### 1.3 Upload Docker Image
Since you can't build directly in portal, use **Azure Cloud Shell** or **Local Docker**:

**Option A: Using Azure Cloud Shell (Built into Portal) - RECOMMENDED**

⚠️ **Important:** Model files are NOT in GitHub (they're too large and in .gitignore). You have 3 options:

**A1. Upload models to Cloud Shell (EASIEST - Use this if you have models trained locally):**
1. On your local machine, zip the models:
```powershell
Compress-Archive -Path models\*.pkl -DestinationPath models.zip
```

2. In Azure Cloud Shell, click **Upload/Download files** icon (⬆️)
3. Upload `models.zip`
4. Extract and build:
```bash
git clone https://github.com/velagalasr/threat-severity-model.git
cd threat-severity-model
unzip ~/models.zip -d models/

# Build Docker image and push to ACR
# If ACR Tasks is not available, use local Docker (Option B below)
az acr build --registry threatmodelacr --image threat-model-api:latest .
```

**A2. Train models in Cloud Shell (if ACR Tasks available):**
1. In Azure Portal, click **Cloud Shell** icon (top right, looks like `>_`)
2. Choose **Bash**
3. Clone repo and train models:
```bash
# Clone your repo
git clone https://github.com/velagalasr/threat-severity-model.git
cd threat-severity-model

# Install dependencies (use --user flag in Cloud Shell)
pip install --user -r requirements.txt

# Add user bin to PATH
export PATH="$HOME/.local/bin:$PATH"

# Add project root to PYTHONPATH (fixes import errors)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Download dataset
python data/download_dataset.py

# Train models (takes 5-10 minutes)
python scripts/train.py

# Now build Docker image with trained models
# Note: Cloud Shell doesn't have Docker daemon, so we use ACR's build service
az acr build --registry threatmodelacr --image threat-model-api:latest .
```

**A2. Upload models to Cloud Shell first:**
1. On your local machine, zip the models:
```powershell
Compress-Archive -Path models/*.pkl -DestinationPath models.zip
```

2. In Azure Cloud Shell, click **Upload/Download files** icon
3. Upload `models.zip`
4. Extract and build:
```bash
git clone https://github.com/velagalasr/threat-severity-model.git
cd threat-severity-model
unzip ~/models.zip -d models/

# Build with existing models
az acr login --name threatmodelacr
az acr build --registry threatmodelacr --image threat-model-api:latest .
```

**A3. Use Azure Blob Storage for models (Production approach):**
1. Upload models to Blob Storage first (see Step 3 below)
2. Modify Dockerfile to download models on startup:
```dockerfile
# Add to Dockerfile before CMD
RUN pip install azure-storage-blob
ENV AZURE_STORAGE_CONNECTION_STRING=""
# Your app will download models from blob storage on startup
```

**Option B: Push from Local Docker Desktop**
```powershell
# In your local terminal (with Docker Desktop installed)
docker build -t threat-model-api:latest .

# Get ACR login credentials from portal
# Portal → Container Registry → Access Keys → Enable Admin User
# Copy username and password

docker login threatmodelacr.azurecr.io
# Enter username and password when prompted

docker tag threat-model-api:latest threatmodelacr.azurecr.io/threat-model-api:latest
docker push threatmodelacr.azurecr.io/threat-model-api:latest
```

### 1.4 Create Container App
1. Search for "**Container Apps**" in Azure Portal
2. Click **+ Create**
3. **Basics tab:**
   - Resource Group: `threat-model-rg`
   - Container app name: `threat-model-api`
   - Region: East US
   - Container Apps Environment: Create new → `threat-model-env`
4. **Container tab:**
   - Image source: Azure Container Registry
   - Registry: `threatmodelacr`
   - Image: `threat-model-api`
   - Tag: `latest`
   - CPU: 1.0
   - Memory: 2.0 Gi
5. **Ingress tab:**
   - Enable Ingress: ✅ Yes
   - Ingress traffic: Accepting traffic from anywhere
   - Target port: `5000`
6. Click **Review + Create** → **Create**

### 1.5 Get Your API URL
1. Go to your Container App → **Overview**
2. Copy the **Application URL** (e.g., `https://threat-model-api.xxxxx.eastus.azurecontainerapps.io`)
3. Test it: `https://YOUR-URL/health`

---

## Step 2: Deploy Web UI to Azure Static Web Apps

### 2.1 Update Web UI with Your API URL
Before deploying, edit `web_ui.html`:
```javascript
// Line ~257, replace:
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:5000'
    : 'https://threat-model-api.xxxxx.eastus.azurecontainerapps.io'; // YOUR CONTAINER APP URL
```

### 2.2 Deploy via GitHub (Automated - EASIEST)
1. Commit and push your changes:
```powershell
git add web_ui.html
git commit -m "Update API URL for Azure deployment"
git push
```

2. In Azure Portal:
   - Search "**Static Web Apps**"
   - Click **+ Create**
   - **Basics:**
     - Resource Group: `threat-model-rg`
     - Name: `threat-model-ui`
     - Plan type: **Free**
     - Region: East US 2
   - **Deployment:**
     - Source: **GitHub**
     - Sign in to GitHub
     - Organization: `velagalasr`
     - Repository: `threat-severity-model`
     - Branch: `main`
   - **Build Details:**
     - Build Presets: Custom
     - App location: `/`
     - Output location: `/`
3. Click **Review + Create** → **Create**

Azure will automatically:
- Create GitHub Actions workflow
- Build and deploy your web UI
- Give you a URL like: `https://threat-model-ui.azurestaticapps.net`

### 2.3 Or Deploy via Azure Storage (Alternative - Manual)
1. Create Storage Account:
   - Search "**Storage accounts**"
   - Create new: `threatmodelui` (unique name)
   - Resource Group: `threat-model-rg`
   - Performance: Standard
   - Redundancy: LRS

2. Enable Static Website:
   - Go to Storage Account → **Static website**
   - Enable: **Enabled**
   - Index document: `index.html`
   - Click **Save**
   - Copy the **Primary endpoint** URL

3. Upload Web UI:
   - Go to **Storage browser** → **Blob containers** → **$web**
   - Click **Upload**
   - Upload `web_ui.html` and rename to `index.html`
   - Set **Blob type:** Block blob
   - Upload

4. Access your UI at the primary endpoint URL

---

## Step 3: Upload Models to Blob Storage (Optional)

### 3.1 Create Storage Account for Models
1. If not created above, create storage account: `threatmodelstorage`
2. Go to **Containers** → **+ Container**
   - Name: `models`
   - Public access level: Private

### 3.2 Upload Model Files
1. Click on `models` container
2. Click **Upload**
3. Select your model files from `models/` folder:
   - `xgboost_model.pkl`
   - `scaler.pkl`
   - `shap_explainer.pkl`
4. Click **Upload**

### 3.3 Update Container App to Use Blob Storage
1. Go to Container App → **Configuration**
2. Add environment variables:
   - `AZURE_STORAGE_CONNECTION_STRING`: Get from Storage Account → Access Keys
   - `MODEL_PATH`: `https://threatmodelstorage.blob.core.windows.net/models/xgboost_model.pkl`
3. Update your `src/config.py` to load from blob storage if env var is set

---

## Step 4: Configure CORS (Security)

### 4.1 Enable CORS on Container App
1. Go to Container App → **CORS**
2. Add allowed origins:
   - `https://threat-model-ui.azurestaticapps.net` (your Static Web App URL)
   - Or `*` for testing (not recommended for production)
3. Allowed methods: `GET, POST`
4. **Save**

---

## Step 5: Monitor Your Application

### 5.1 Add Application Insights
1. Search "**Application Insights**"
2. Create new:
   - Name: `threat-model-insights`
   - Resource Group: `threat-model-rg`
3. Go to Container App → **Application Insights**
4. Enable → Select your insights instance

### 5.2 View Metrics
- Container App → **Metrics**: CPU, Memory, Request count
- Application Insights → **Live Metrics**: Real-time requests
- **Logs**: View errors and traces

---

## Quick Reference URLs

After deployment, you'll have:
- **API:** `https://threat-model-api.xxxxx.eastus.azurecontainerapps.io`
- **Web UI:** `https://threat-model-ui.azurestaticapps.net`
- **Models:** `https://threatmodelstorage.blob.core.windows.net/models/`

### Test Your Deployment:
```powershell
# Test API health
curl https://YOUR-CONTAINER-APP-URL/health

# Test Web UI
# Open in browser: https://YOUR-STATIC-WEB-APP-URL
```

---

## Costs Summary (Monthly)

| Service | Cost | Notes |
|---------|------|-------|
| Container App | $15-30 | 1 instance, pay-per-use |
| Static Web App | **$0** | Free tier |
| Container Registry | $5 | Basic SKU |
| Blob Storage | $1-2 | Standard LRS |
| Application Insights | $5-10 | First 5GB free |
| **Total** | **~$26-47/month** | |

**First month:** Often $0 with Azure free credits!

---

## Troubleshooting

### Container App won't start
- Check logs: Container App → **Log stream**
- Verify image: Container Registry → Repositories
- Check port: Must be 5000 (matches Dockerfile EXPOSE)

### Web UI can't connect to API
- Verify API URL in web_ui.html
- Check CORS settings on Container App
- Test API directly: `curl https://YOUR-API-URL/health`

### Models not found
- Verify files uploaded to Blob Storage
- Check environment variables in Container App
- Ensure connection string is correct

---

## Next Steps

1. ✅ **Custom Domain**: Add your own domain to Static Web App
2. ✅ **SSL Certificate**: Free with Static Web Apps
3. ✅ **Authentication**: Add Azure AD authentication
4. ✅ **CI/CD**: GitHub Actions auto-deploy on every push
5. ✅ **Scaling**: Configure auto-scaling rules

---

**No CLI needed! Everything done through Azure Portal** ✨

**Estimated Time:** 30-45 minutes
**Cost:** ~$26-47/month (or $0 first month with free credits)

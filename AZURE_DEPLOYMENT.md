# Azure Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         USERS                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──────────────────┬─────────────────────┐
                     │                  │                     │
              ┌──────▼──────┐    ┌─────▼─────┐       ┌──────▼──────┐
              │   Web UI    │    │  API      │       │   Models    │
              │   Static    │    │ Container │       │   Blob      │
              │   Web App   │    │   Apps    │       │  Storage    │
              └─────────────┘    └───────────┘       └─────────────┘
```

## Option 1: Azure Container Apps + Static Web Apps (RECOMMENDED)

### Prerequisites

**Install Azure CLI (choose one method):**

```powershell
# Method 1: Using winget (Windows Package Manager)
winget install Microsoft.AzureCLI

# Method 2: Using MSI installer (if winget not available)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi
Start-Process msiexec.exe -ArgumentList '/I AzureCLI.msi /quiet' -Wait
Remove-Item .\AzureCLI.msi

# Method 3: Using Chocolatey
choco install azure-cli
```

**After installation, restart your terminal and verify:**
```powershell
az --version
```

**Login to Azure:**
```powershell
# Login
az login

# Set subscription
az account set --subscription "YOUR_SUBSCRIPTION_NAME"

# Verify
az account show
```

### Step 1: Deploy API to Azure Container Apps

```powershell
# Variables
$RESOURCE_GROUP="threat-model-rg"
$LOCATION="eastus"
$CONTAINER_APP_ENV="threat-model-env"
$CONTAINER_APP_NAME="threat-model-api"
$ACR_NAME="threatmodelacr$(Get-Random -Minimum 1000 -Maximum 9999)"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic

# Build and push Docker image
az acr build --registry $ACR_NAME --image threat-model-api:latest .

# Create Container Apps environment
az containerapp env create `
  --name $CONTAINER_APP_ENV `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION

# Deploy container app
az containerapp create `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --environment $CONTAINER_APP_ENV `
  --image "$ACR_NAME.azurecr.io/threat-model-api:latest" `
  --target-port 5000 `
  --ingress external `
  --registry-server "$ACR_NAME.azurecr.io" `
  --cpu 1.0 --memory 2.0Gi `
  --min-replicas 1 --max-replicas 5

# Get API URL
$API_URL = az containerapp show `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "properties.configuration.ingress.fqdn" -o tsv

Write-Host "API deployed at: https://$API_URL"
```

### Step 2: Deploy Web UI to Azure Static Web Apps

```powershell
# Update web_ui.html with your API URL first
# Edit line: const API_URL = 'https://YOUR-API-URL';

# Install Static Web Apps CLI
npm install -g @azure/static-web-apps-cli

# Deploy using Azure CLI
az staticwebapp create `
  --name "threat-model-ui" `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --source "." `
  --app-location "/" `
  --output-location "."

# Or deploy via GitHub Actions (automated)
# See: https://docs.microsoft.com/azure/static-web-apps/deploy-nextjs
```

### Step 3: Upload Models to Azure Blob Storage (Optional)

```powershell
# Create storage account
$STORAGE_ACCOUNT="threatmodelstorage$(Get-Random -Minimum 1000 -Maximum 9999)"

az storage account create `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --sku Standard_LRS

# Create container
az storage container create `
  --name models `
  --account-name $STORAGE_ACCOUNT `
  --public-access off

# Upload models
az storage blob upload-batch `
  --account-name $STORAGE_ACCOUNT `
  --destination models `
  --source ./models `
  --pattern "*.pkl"

# Get connection string and update your app
$CONN_STRING = az storage account show-connection-string `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --query connectionString -o tsv

Write-Host "Add this to Container App environment variables:"
Write-Host "AZURE_STORAGE_CONNECTION_STRING=$CONN_STRING"
```

## Option 2: Azure App Service (Simpler)

```powershell
# Create App Service Plan
az appservice plan create `
  --name "threat-model-plan" `
  --resource-group $RESOURCE_GROUP `
  --location $LOCATION `
  --is-linux `
  --sku B2

# Create Web App
az webapp create `
  --name "threat-model-api-$(Get-Random -Minimum 1000 -Maximum 9999)" `
  --resource-group $RESOURCE_GROUP `
  --plan "threat-model-plan" `
  --deployment-container-image-name "$ACR_NAME.azurecr.io/threat-model-api:latest"

# Configure Web App
az webapp config appsettings set `
  --name "threat-model-api" `
  --resource-group $RESOURCE_GROUP `
  --settings WEBSITES_PORT=5000

# Enable container registry integration
az webapp config container set `
  --name "threat-model-api" `
  --resource-group $RESOURCE_GROUP `
  --docker-custom-image-name "$ACR_NAME.azurecr.io/threat-model-api:latest" `
  --docker-registry-server-url "https://$ACR_NAME.azurecr.io"
```

## Option 3: Host Web UI in Blob Storage (Cheapest)

```powershell
# Enable static website hosting
az storage blob service-properties update `
  --account-name $STORAGE_ACCOUNT `
  --static-website `
  --index-document web_ui.html

# Upload web UI
az storage blob upload `
  --account-name $STORAGE_ACCOUNT `
  --container-name '$web' `
  --file web_ui.html `
  --name index.html

# Get website URL
$WEB_URL = az storage account show `
  --name $STORAGE_ACCOUNT `
  --resource-group $RESOURCE_GROUP `
  --query "primaryEndpoints.web" -o tsv

Write-Host "Web UI available at: $WEB_URL"
```

## Update Web UI for Azure

Before deploying, update `web_ui.html`:

```javascript
// Replace this line:
const API_URL = 'http://127.0.0.1:5000';

// With your Azure API URL:
const API_URL = 'https://threat-model-api-xxxx.azurecontainerapps.io';
```

## Cost Estimates (Monthly)

### Development/Testing
- Container Apps (1 instance): ~$15-30
- Static Web Apps (Free tier): $0
- Blob Storage: ~$1-5
- **Total: ~$16-35/month**

### Production
- Container Apps (1-5 instances with autoscale): ~$50-150
- Static Web Apps (Standard): $9
- Blob Storage: ~$5-10
- Application Insights: ~$5-20
- **Total: ~$69-189/month**

## Monitoring & Logging

```powershell
# Create Application Insights
az monitor app-insights component create `
  --app "threat-model-insights" `
  --location $LOCATION `
  --resource-group $RESOURCE_GROUP `
  --application-type web

# Get instrumentation key
$INSTRUMENTATION_KEY = az monitor app-insights component show `
  --app "threat-model-insights" `
  --resource-group $RESOURCE_GROUP `
  --query instrumentationKey -o tsv

# Add to Container App
az containerapp update `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --set-env-vars "APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=$INSTRUMENTATION_KEY"
```

## CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Azure
      uses: azure/login@v1
      with:
        creds: ${{ secrets.AZURE_CREDENTIALS }}
    
    - name: Build and push Docker image
      run: |
        az acr build --registry ${{ secrets.ACR_NAME }} \
          --image threat-model-api:${{ github.sha }} \
          --image threat-model-api:latest .
    
    - name: Deploy to Container Apps
      run: |
        az containerapp update \
          --name threat-model-api \
          --resource-group threat-model-rg \
          --image ${{ secrets.ACR_NAME }}.azurecr.io/threat-model-api:${{ github.sha }}
```

## Security Best Practices

1. **Enable HTTPS only** (automatically enabled on Container Apps)
2. **Use Managed Identity** for Azure resource access
3. **Store secrets in Azure Key Vault**
4. **Enable CORS** only for your Static Web App domain
5. **Use Azure Front Door** for DDoS protection
6. **Enable Application Gateway WAF** for production

```powershell
# Example: Add CORS to Container App
az containerapp ingress cors enable `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --allowed-origins "https://YOUR-STATIC-WEB-APP.azurestaticapps.net" `
  --allowed-methods GET POST `
  --allowed-headers "*"
```

## Testing Your Deployment

```powershell
# Test API health
curl https://YOUR-API-URL/health

# Test prediction
curl -X POST https://YOUR-API-URL/predict `
  -H "Content-Type: application/json" `
  -d "@test_request.json"
```

## Troubleshooting

### Container App logs
```powershell
az containerapp logs show `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --follow
```

### Check container status
```powershell
az containerapp show `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --query "properties.runningStatus"
```

### Scale manually
```powershell
az containerapp update `
  --name $CONTAINER_APP_NAME `
  --resource-group $RESOURCE_GROUP `
  --min-replicas 2 --max-replicas 10
```

## Next Steps

1. Set up **Azure Monitor alerts** for API failures
2. Enable **Auto-scaling rules** based on CPU/memory
3. Configure **Azure Front Door** for global distribution
4. Implement **Azure API Management** for rate limiting
5. Use **Azure Machine Learning** for model retraining pipeline

---

**Estimated Setup Time:** 20-30 minutes
**Difficulty:** Intermediate
**Cost:** Starting at $16/month (dev) to $69-189/month (production)

# Deployment Guide 🚀

This guide explains how to deploy the Multi-Modal Document Intelligence Platform using a hybrid approach: **Vercel** for the frontend and **Render** for the backend.

---

## 🏗️ 1. Backend Deployment (Render)

We will deploy the FastAPI backend as a Docker container on Render.

### Prerequisites
- A [Render](https://render.com/) account.
- Your project pushed to GitHub.

### Steps
1.  **New Web Service**: Go to Render Dashboard → New → Web Service.
2.  **Connect Repo**: Select your `ai_agent_builder` repository.
3.  **Configuration**:
    - **Name**: `ai-agent-backend` (or similar)
    - **Language**: Docker
    - **Region**: Choose one close to you (e.g., Frankfurt, Oregon).
    - **Branch**: `main`
    - **Root Directory**: `.` (Leave empty or default)
    - **Dockerfile Path**: `./docker/Dockerfile.backend`  **(Crucial!)**
4.  **Environment Variables**:
    Add the following variables (copy values from your local `.env`, but use production secrets):
    - `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)
    - `QDRANT_HOST`: If using cloud Qdrant, put the URL here. If using local Qdrant in Docker, Render doesn't support docker-compose directly in Web Services.
        - **Option A (Recommended)**: Use [Qdrant Cloud](https://qdrant.tech/) (Free Tier available) and set `QDRANT_HOST` and `QDRANT_API_KEY`.
        - **Option B**: Deploy Qdrant as a separate generic service on Render (more complex).
    - `TRANSFORMERS_OFFLINE`: `0` (Allow downloading models)
    - `HF_HUB_OFFLINE`: `0`
5.  **Instance Type**: Select "Starter" or higher (Free tier might be too slow for ML models).
6.  **Deploy**: Click "Create Web Service".

**Note Backend URL**: Once deployed, Render will give you a URL like `https://ai-agent-backend.onrender.com`.

---

## ⚛️ 2. Frontend Deployment (Vercel)

We will deploy the React frontend on Vercel.

### Steps
1.  **New Project**: Go to Vercel Dashboard → Add New → Project.
2.  **Import Repo**: Select `ai_agent_builder`.
3.  **Configure Project**:
    - **Framework Preset**: Vite
    - **Root Directory**: `frontend` **(Important: Click Edit and select `frontend` folder)**
4.  **Build & Output Settings**:
    - Build Command: `npm run build`
    - Output Directory: `dist`
    - Install Command: `npm install`
5.  **Environment Variables**:
    - Key: `VITE_API_BASE_URL`
    - Value: `https://ai-agent-backend.onrender.com` (The URL from Step 1)
6.  **Deploy**: Click "Deploy".

---

## 🔄 3. Final Verification

1.  Open your Vercel URL (e.g., `https://ai-agent-frontend.vercel.app`).
2.  Check the browser console to ensure requests are going to your Render backend (not localhost).
3.  Upload a document and verify the flow.

### Troubleshooting
- **CORS Error**: If the frontend gets CORS errors, go to your Backend code (`backend/config.py`) or environment variables on Render and update `CORS_ORIGINS` to include your new Vercel domain.
    - Render Env Var: `CORS_ORIGINS` = `https://ai-agent-frontend.vercel.app`

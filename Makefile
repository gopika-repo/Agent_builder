# Multi-Modal Document Intelligence Platform
# Development & Deployment Commands

.PHONY: help install dev test lint docker up down clean

# Default target
help:
	@echo "╔═══════════════════════════════════════════════════════════════╗"
	@echo "║       Multi-Modal Document Intelligence Platform             ║"
	@echo "╠═══════════════════════════════════════════════════════════════╣"
	@echo "║ DEVELOPMENT                                                   ║"
	@echo "║   make install     - Install all dependencies                ║"
	@echo "║   make dev         - Start development servers               ║"
	@echo "║   make test        - Run all tests                           ║"
	@echo "║   make lint        - Run linting                             ║"
	@echo "║                                                               ║"
	@echo "║ DOCKER                                                        ║"
	@echo "║   make docker      - Build Docker images                     ║"
	@echo "║   make up          - Start with Docker Compose               ║"
	@echo "║   make down        - Stop Docker Compose                     ║"
	@echo "║   make logs        - View Docker logs                        ║"
	@echo "║                                                               ║"
	@echo "║ CI/CD                                                         ║"
	@echo "║   make ci          - Run CI pipeline locally                 ║"
	@echo "║   make release     - Create a new release                    ║"
	@echo "╚═══════════════════════════════════════════════════════════════╝"

# ============================================
# Installation
# ============================================
install: install-backend install-frontend install-hooks
	@echo "✅ All dependencies installed!"

install-backend:
	@echo "📦 Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	cd backend && pip install pytest pytest-cov pytest-asyncio ruff mypy

install-frontend:
	@echo "📦 Installing frontend dependencies..."
	cd frontend && npm ci

install-hooks:
	@echo "🪝 Installing pre-commit hooks..."
	pip install pre-commit
	pre-commit install

# ============================================
# Development
# ============================================
dev: dev-services
	@echo "🚀 Starting development environment..."
	@make -j2 dev-backend dev-frontend

dev-backend:
	@echo "🐍 Starting backend..."
	cd backend && uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "⚛️ Starting frontend..."
	cd frontend && npm run dev

dev-services:
	@echo "🗄️ Starting Qdrant..."
	docker run -d --name qdrant-dev -p 6333:6333 qdrant/qdrant:latest || true

# ============================================
# Testing
# ============================================
test: test-backend test-frontend
	@echo "✅ All tests passed!"

test-backend:
	@echo "🧪 Running backend tests..."
	cd backend && pytest tests/ -v --cov=. --cov-report=term-missing

test-frontend:
	@echo "🧪 Running frontend tests..."
	cd frontend && npm test -- --run

test-integration:
	@echo "🔗 Running integration tests..."
	docker-compose up -d
	sleep 10
	curl -f http://localhost:8000/health
	docker-compose down

# ============================================
# Linting & Formatting
# ============================================
lint: lint-backend lint-frontend
	@echo "✅ Linting complete!"

lint-backend:
	@echo "🔍 Linting backend..."
	cd backend && ruff check .
	cd backend && mypy --ignore-missing-imports .

lint-frontend:
	@echo "🔍 Linting frontend..."
	cd frontend && npm run lint

format:
	@echo "🎨 Formatting code..."
	cd backend && ruff format .
	cd frontend && npm run format || true

# ============================================
# Docker
# ============================================
docker: docker-backend docker-frontend
	@echo "✅ Docker images built!"

docker-backend:
	@echo "🐳 Building backend image..."
	docker build -t doc-intel-backend:latest -f docker/Dockerfile.backend .

docker-frontend:
	@echo "🐳 Building frontend image..."
	docker build -t doc-intel-frontend:latest -f docker/Dockerfile.frontend .

up:
	@echo "🚀 Starting services..."
	docker-compose up -d
	@echo "✅ Services running!"
	@echo "   Backend: http://localhost:8000"
	@echo "   Frontend: http://localhost:3000"
	@echo "   Qdrant: http://localhost:6333"

down:
	@echo "🛑 Stopping services..."
	docker-compose down

logs:
	docker-compose logs -f

# ============================================
# CI/CD
# ============================================
ci: lint test docker
	@echo "✅ CI pipeline passed!"

release:
	@read -p "Enter version (e.g., v1.0.0): " version; \
	git tag -a $$version -m "Release $$version"; \
	git push origin $$version; \
	echo "✅ Release $$version created and pushed!"

# ============================================
# Cleanup
# ============================================
clean:
	@echo "🧹 Cleaning up..."
	rm -rf backend/__pycache__
	rm -rf backend/**/__pycache__
	rm -rf backend/.pytest_cache
	rm -rf backend/.coverage
	rm -rf backend/htmlcov
	rm -rf frontend/node_modules
	rm -rf frontend/dist
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup complete!"

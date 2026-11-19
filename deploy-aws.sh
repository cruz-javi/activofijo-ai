#!/usr/bin/env bash
set -e

echo "=== 1. Ajustando permisos en ~/app ==="
sudo chown -R ubuntu:ubuntu ~/app

echo "=== 2. Actualizando activofijo-core, activofijo-web y activofijo-ai a origin/main ==="
git -C ~/app/activofijo-core fetch origin main
git -C ~/app/activofijo-core checkout main
git -C ~/app/activofijo-core reset --hard origin/main

git -C ~/app/activofijo-web fetch origin main
git -C ~/app/activofijo-web checkout main
git -C ~/app/activofijo-web reset --hard origin/main

git -C ~/app/activofijo-ai fetch origin main
git -C ~/app/activofijo-ai checkout main
git -C ~/app/activofijo-ai reset --hard origin/main

echo "=== 3. Instalando docker-compose.yml maestro en ~/app ==="
cp ~/app/activofijo-ai/docker-compose.prod.yml ~/app/docker-compose.yml

if [ -f ~/app/.env ]; then
  sed -i 's/^DATA_SOURCE_MODE=fresh/DATA_SOURCE_MODE=hybrid/' ~/app/.env || true
  grep -q '^DATA_SOURCE_MODE=' ~/app/.env || echo 'DATA_SOURCE_MODE=hybrid' >> ~/app/.env
  grep -q '^AI_SERVICE_ENABLED=' ~/app/.env || echo 'AI_SERVICE_ENABLED=true' >> ~/app/.env
  sed -i 's/^AI_SERVICE_ENABLED=false/AI_SERVICE_ENABLED=true/' ~/app/.env || true
  grep -q '^AI_SERVICE_URL=' ~/app/.env || echo 'AI_SERVICE_URL=http://activofijo-ai:8000' >> ~/app/.env
  grep -q '^AI_INTERNAL_API_KEY=' ~/app/.env || echo 'AI_INTERNAL_API_KEY=uagrm_internal_ai_bridge_secret_key_2026_activo_fijo_seguro' >> ~/app/.env
  grep -q '^AI_REQUEST_TIMEOUT_MS=' ~/app/.env || echo 'AI_REQUEST_TIMEOUT_MS=165000' >> ~/app/.env
  grep -q '^LLM_PROVIDER=' ~/app/.env || echo 'LLM_PROVIDER=groq' >> ~/app/.env
  sed -i 's/^LLM_PROVIDER=ollama/LLM_PROVIDER=groq/' ~/app/.env || true
  grep -q '^LLM_BASE_URL=' ~/app/.env || echo 'LLM_BASE_URL=https://api.groq.com/openai/v1' >> ~/app/.env
  sed -i 's|^LLM_BASE_URL=http://host.docker.internal:11434/v1|LLM_BASE_URL=https://api.groq.com/openai/v1|' ~/app/.env || true
  grep -q '^LLM_MODEL=' ~/app/.env || echo 'LLM_MODEL=qwen/qwen3.8-27b' >> ~/app/.env
  sed -i 's|^LLM_MODEL=uagrm-activo-fijo|LLM_MODEL=qwen/qwen3.8-27b|' ~/app/.env || true
  if ! grep -q '^LLM_API_KEY=gsk_' ~/app/.env; then
    FB_KEY=$(grep '^LLM_FALLBACK_API_KEY=' ~/app/.env | cut -d'=' -f2- || true)
    if [ -n "$FB_KEY" ]; then
      grep -q '^LLM_API_KEY=' ~/app/.env && sed -i "s|^LLM_API_KEY=.*|LLM_API_KEY=$FB_KEY|" ~/app/.env || echo "LLM_API_KEY=$FB_KEY" >> ~/app/.env
    fi
  fi
fi

echo "=== 4. Compilando y levantando contenedores secuencialmente ==="
cd ~/app
docker compose build core
docker compose up -d --force-recreate core

docker compose build ai
docker compose up -d --force-recreate ai

docker compose build web
docker compose up -d --force-recreate web

docker image prune -f
docker compose ps

echo "=== 5. Verificando salud ==="
sleep 4
echo -n "Core: " && curl -s http://127.0.0.1:3000/health && echo ""
echo -n "AI:   " && curl -s http://127.0.0.1:8000/health && echo ""
echo "Web:" && curl -I -s http://127.0.0.1:3001 | head -n 5

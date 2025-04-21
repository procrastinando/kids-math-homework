# README.md

## Overview
This repository contains a Gradio-based application (`julia.py`) for poem pronunciation practice and mathematics learning. The app auto-installs its Python dependencies on startup and exposes port `99` for access.

---

## Prerequisites
Make sure you have Docker installed on your system. If not, you can install it with:

```bash
sudo apt install -y curl git
curl -fsSL https://get.docker.com | sudo sh
```

---

## Build the Docker Image
To build the Docker image directly from the GitHub repository (from the `main` branch), run:

```bash
docker build -t julia https://github.com/procrastinando/kids-math-homework.git#main:.
```

- `-t julia` tags the image as `julia`.

---

## Run the Container
Once built, you can launch the container:

```bash
docker run --p 666:666 julia
```

- `-p 666:666` maps port 666 of the container to port 666 on the host.

---

## Updating to a New Version
When you push updates to the `main` branch, rebuild and redeploy:

```bash
docker build -t julia https://github.com/procrastinando/kids-math-homework.git#main:.

docker stop julia_container || true
docker rm julia_container  || true

docker run -d --name julia_container -p 666:66 julia
```

1. **Rebuild** the image from the updated repository.
2. **Stop** and **remove** the old container (if running).
3. **Run** a new container from the updated image.

---

## Docker Compose
To streamline container deployment, you can use Docker Compose. Create a file named `docker-compose.yml` with the following content:

```yaml
services:
  julia-app:
    image: julia
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "666:666"
    restart: unless-stopped
```

# Kids math homework
A simple mathematic homework for kids using gradio webui

### Requirements:
```
sudo apt install -y curl git
curl -fsSL https://get.docker.com | sudo sh
```6
### Manual Installation:
Place the `app.py` file in `/root/math` directory. 
```
docker run -d --name math --restart always --network host -v /root/math:/root/math python:3.12-alpine sh -c "tail -f /dev/null & pip install --no-cache-dir -r /root/math/requirements.txt && python /root/math/app.py"
```
### Build the image:
- Locally: `docker build -t math .`
- From the repository: `docker build -t math https://github.com/procrastinando/kids-math-homework.git#main:.`
- To update:
```
docker rm -f math
docker build --no-cache --pull -t math https://github.com/procrastinando/kids-math-homework.git#main:.
```
### Option 1: Run by command:
```
docker run -d --name math -p 666:8501 --restart unless-stopped
```
### Option 2: Run it as stack:
```
services:
  math:
    container_name: math
    image: math
    network: host
    restart: unless-stopped
```

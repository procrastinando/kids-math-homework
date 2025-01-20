# Use the Python 3.12 Alpine base image
FROM python:3.12-alpine

# Set the working directory
WORKDIR /app

# Install nano and other necessary dependencies
RUN apk add --no-cache \
    nano \
    git

# Clone the repository containing app.py
RUN git clone https://github.com/procrastinando/kids-math-homework.git /app

# Install Python dependencies
RUN pip install --no-cache-dir gradio

# Expose the port used by the Gradio app (adjust if necessary)
EXPOSE 666

# Command to run the application
CMD ["python", "app.py"]
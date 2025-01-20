# Use the Python 3.12 Alpine base image
FROM python:3.12-alpine

# Set the working directory
WORKDIR /app

# Install wget (or curl)
RUN apk add --no-cache wget

# Download the app.py file
RUN wget https://raw.githubusercontent.com/procrastinando/kids-math-homework/main/app.py -O /app/app.py

# Install Python dependencies
RUN pip install --no-cache-dir gradio

# Expose the port used by the Gradio app
EXPOSE 666

# Command to run the application
CMD ["python", "app.py"]

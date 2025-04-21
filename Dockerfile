FROM python:3.12-alpine

WORKDIR /julia

# Copy application files
COPY julia.py /julia/

# Expose application port
EXPOSE 666

# Launch the script (it will install needed packages on startup)
CMD ["python", "julia.py"]

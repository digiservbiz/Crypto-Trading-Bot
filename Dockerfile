# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r scripts/training/requirements.txt

# Fix pandas-ta bug
RUN sed -i 's/NaN/nan/g' /usr/local/lib/python3.11/site-packages/pandas_ta/momentum/squeeze_pro.py

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV PYTHONPATH /app

# Run app.py when the container launches
CMD ["streamlit", "run", "scripts/app.py"]

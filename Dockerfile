# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirement.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (if applicable)
EXPOSE 8000

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Use the entrypoint script to run the application
ENTRYPOINT ["/app/entrypoint.sh"]
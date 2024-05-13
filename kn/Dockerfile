# Use Ubuntu as the base image
FROM ubuntu:latest

# Update the package lists and install Python and other required dependencies
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev build-essential && \
    apt-get install -y libpq-dev && \
    apt-get install -y libsm6 libxext6 libxrender-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install the required Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Copy the source code to the working directory
COPY . /app/

# Set environment variables, if needed
# ENV VARIABLE_NAME value

# Expose the port on which your Django server will run
EXPOSE 8000

# Start the Django server
#CMD ["python3", "main.py", "runserver", "0.0.0.0:8000"]
CMD ["python3", "main.py"]


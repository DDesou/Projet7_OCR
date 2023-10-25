# Choose our version of Python
FROM python:3.10.11

# Set up a working directory
WORKDIR /code

# Copy just the requirements into the working directory so it gets cached by itself
COPY requirements.txt .

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the code into the working directory
COPY . .

EXPOSE 8501

CMD streamlit run dashboard.py
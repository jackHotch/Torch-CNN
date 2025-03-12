FROM python:latest

WORKDIR /train

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "mlp.py"]
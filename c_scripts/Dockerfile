FROM python:3.8

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "/github/workspace/script.py" ]
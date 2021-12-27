FROM python:3.8

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /github/workspace

CMD [ "python", "/github/workspace/c_scripts/script.py" ]
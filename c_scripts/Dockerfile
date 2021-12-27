FROM python:3.8

COPY /github/workspace/3_scripts/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "/github/workspace/3_scripts/script.py" ]
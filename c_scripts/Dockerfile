FROM python:3.8

COPY /c_scripts/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "/github/workspace/c_scripts/script.py" ]
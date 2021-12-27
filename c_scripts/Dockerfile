FROM python:3.8

COPY /github/workspace/c_scripts/requirements.txt ./requirements
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "/github/workspace/c_scripts/script.py" ]
FROM python:3.11.7
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt
EXPOSE 3000
CMD python ./start.py
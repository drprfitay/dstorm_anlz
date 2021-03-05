FROM python:3.8

RUN apt update && apt install -y python3-pip
RUN pip3 install --upgrade pip
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt 
EXPOSE 5000
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV FLASK_APP main.py
ENV FLASK_ENV development
COPY ./ .
entrypoint ["python3"]
CMD ["main.py"]
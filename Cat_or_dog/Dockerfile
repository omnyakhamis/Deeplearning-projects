# DOCKERfile have to be uppercase without extention
# open cmd - desactivate conda if it is activate - exit from base enviroment
#pipenv # we did this comand ti install the API - because we used maxim API

#how we build file docker from scratch - for our project (titanic)
# create docker file - just file
From  python:3.7.3-stretch

RUN mkdir /Cat_or_dog
# create working directory


WORKDIR /Cat_or_dog
# tell docker that this is working directory

COPY requirements.txt /Cat_or_dog/
# i will tell docker to copy these staf
# if we donot have cmd run pipenv
# pip freez > requirements.txt

RUN pip install -r requirements.txt
# we install theses requirements in the working directory

COPY . /Cat_or_dog/
# ask docker to copy every this from my folder application

#EXPOSE 5000
EXPOSE $PORT
# determin the port

#CMD python train.py
# ask docker to run this file
CMD python run.py

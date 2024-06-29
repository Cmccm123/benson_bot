from pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

COPY . benson_bot

WORKDIR benson_bot

RUN python3 -m pip install -r requirement.txt

RUN python3 script/bensonMsgToDB.py
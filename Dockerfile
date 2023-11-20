FROM  nvcr.io/nvidia/tensorflow:21.10-tf2-py3
# maybe we also have a requirements.txt file
COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt
COPY . /workspace/project/
ENTRYPOINT ["python3"]
CMD ["/workspace/project/train.py"] 
FROM python:3.10

WORKDIR /app

COPY app.py /app
COPY requirements.txt /app
COPY KserveML.py /app

RUN pip install -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder='/tmp')"
EXPOSE 8501
 
ENTRYPOINT [ "streamlit","run","app.py" ]
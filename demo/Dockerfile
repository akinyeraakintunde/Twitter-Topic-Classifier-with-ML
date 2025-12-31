FROM python:3.11-slim

WORKDIR /app

# Copy demo code
COPY demo/ /app/demo/

# Copy model folder (you must have models/model.joblib in repo root)
COPY models/ /app/models/

RUN pip install --no-cache-dir -r /app/demo/requirements.txt

ENV PORT=10000
EXPOSE 10000

CMD ["bash", "-lc", "streamlit run /app/demo/app.py --server.address 0.0.0.0 --server.port ${PORT}"]
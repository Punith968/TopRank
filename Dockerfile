FROM python:3.10-slim

RUN useradd -m -u 1000 user
RUN mkdir -p /app && chown -R user:user /app
WORKDIR /app

USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

EXPOSE 7860

CMD ["python", "app.py"]

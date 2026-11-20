FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
ARG EXTRAS=""
RUN pip install --no-cache-dir -e ".${EXTRAS}"
COPY . .
CMD ["uvicorn", "claims_agent.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

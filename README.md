# Semantic Search Backend API

Semantic search API using FastAPI and transformer models for product search.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

### 1. Create a virtual environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Launch

### 1. Make sure the virtual environment is activated

```bash
source venv/bin/activate  # On macOS/Linux
# or venv\Scripts\activate on Windows
```

### 2. Start the server

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000

## Endpoints

- **GET /** : API status
- **POST /search** : Semantic search
- **GET /products** : List all products
- **GET /docs** : Automatic Swagger documentation

## Quick test

```bash
# Test search endpoint
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "summer clothing"}'

# Test products endpoint
curl -X GET "http://localhost:8000/products"

# Test status
curl -X GET "http://localhost:8000/"
```
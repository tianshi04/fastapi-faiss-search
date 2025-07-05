# 🚀 FastAPI Vector Search Service

A simple and efficient API service for storing and searching vector embeddings, built with **FastAPI** and **FAISS**.

This service is designed for rapid deployment of AI-powered features like image recognition, semantic search, or recommendation systems. It's lightweight enough to run on edge devices like a Raspberry Pi.

## ✨ Features

- **🔌 Simple API:** Clean endpoints to `add` and `search` vectors.
- **⚡ High-Speed Search:** Leverages FAISS for in-memory, blazing-fast vector similarity searches.
- **💾 Persistent Storage:** Automatically saves the index to disk, so your data survives a server restart.
- **⚙️ Configurable:** Easily set key parameters (like vector dimension) via environment variables without changing the code.
- **🪄 Automatic API Docs:** Interactive API documentation (Swagger UI & ReDoc) provided out-of-the-box by FastAPI.

## 🚀 Getting Started

### 1. ✅ Prerequisites

- Python 3.8+
- A virtual environment (recommended)

### 2. 🧰 Installation

1. **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 3. 🔧 Configuration

1. Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

2. Edit the `.env` file to match your project's requirements, especially `VECTOR_DIMENSION`.

### 4. ▶️ Running the Service

Start the API server with the following command:

```bash
python main.py
```

The server will be running at `http://localhost:8000`.

## 📡 API Usage

Once the server is running, you can access the interactive API documentation at `http://localhost:8000/docs`.

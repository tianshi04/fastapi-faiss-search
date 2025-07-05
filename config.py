# config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Class để quản lý tất cả cấu hình của ứng dụng.
    """
    
    # --- Cấu hình FAISS & Dữ liệu ---
    VECTOR_DIMENSION: int = 128
    DATA_DIR: str = "data"
    INDEX_FILE: str = "product_database.index"
    IDS_FILE: str = "product_ids.json"
    
    # --- Cấu hình API Server (Uvicorn) ---
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    
    # --- Cấu hình Logic Tìm kiếm ---
    # Giá trị này dùng để chuyển đổi khoảng cách L2 (distance) thành điểm tin cậy (confidence).
    # Công thức: confidence = max(0, 1.0 - (distance / NORMALIZATION_FACTOR))
    # Nếu vector của bạn được chuẩn hóa L2, khoảng cách max là 2.0.
    DISTANCE_NORMALIZATION_FACTOR: float = 2.0

    class Config:
        # Pydantic sẽ tìm file .env và tải các biến từ đó
        # Điều này rất hữu ích cho việc phát triển cục bộ
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Tạo một instance duy nhất của Settings để import vào các module khác
settings = Settings()
# main.py

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List
import uvicorn

# Import class handler từ file faiss_handler.py
from faiss_handler import FaissHandler
from config import settings

# --- Pydantic Models để validate dữ liệu đầu vào ---

class VectorInput(BaseModel):
    product_id: int = Field(..., description="ID của sản phẩm, liên kết với CSDL quan hệ.")
    vector: List[float] = Field(..., description="Vector đặc trưng của sản phẩm.")

class SearchInput(BaseModel):
    vector: List[float] = Field(..., description="Vector cần tìm kiếm.")
    k: int = Field(default=1, gt=0, description="Số lượng kết quả gần nhất muốn trả về.")

# --- Khởi tạo ứng dụng FastAPI và FaissHandler ---

app = FastAPI(
    title="SmartCart Vector Search API",
    description="API để thêm và tìm kiếm vector sản phẩm sử dụng FAISS.",
    version="1.0.0"
)

# Tạo một instance duy nhất của FaissHandler
# Instance này sẽ được chia sẻ giữa các request
db_handler = FaissHandler()


# --- Định nghĩa các API Endpoints ---

@app.get("/", summary="Kiểm tra trạng thái dịch vụ")
def read_root():
    """Endpoint cơ bản để kiểm tra xem dịch vụ có đang chạy không."""
    return {
        "status": "Service is running",
        "total_vectors": db_handler.get_total_vectors()
    }

@app.post("/add", summary="Thêm một vector sản phẩm mới")
def add_vector(item: VectorInput):
    """
    Nhận một `product_id` và `vector`, thêm vào cơ sở dữ liệu FAISS.
    Dữ liệu sẽ được lưu tự động.
    """
    if len(item.vector) != db_handler.dimension:
        raise HTTPException(
            status_code=400, 
            detail=f"Vector dimension mismatch. Expected {db_handler.dimension}, got {len(item.vector)}."
        )
    
    try:
        total_vectors = db_handler.add(item.product_id, item.vector)
        return {
            "message": "Vector added successfully",
            "product_id": item.product_id,
            "total_vectors_in_db": total_vectors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", summary="Tìm kiếm sản phẩm theo vector")
def search_vector(item: SearchInput):
    """
    Nhận một `vector` và tìm kiếm `k` sản phẩm gần nhất.
    Trả về danh sách các `(product_id, confidence_score)`.
    """
    if db_handler.get_total_vectors() == 0:
        return {"message": "Database is empty. No results.", "results": []}

    if len(item.vector) != db_handler.dimension:
        raise HTTPException(
            status_code=400, 
            detail=f"Vector dimension mismatch. Expected {db_handler.dimension}, got {len(item.vector)}."
        )
    
    try:
        results = db_handler.search(item.vector, item.k)
        return {
            "message": f"Found {len(results)} results.",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear_all", summary="Xóa toàn bộ dữ liệu")
def clear_all_data():
    """
    Endpoint nguy hiểm: Xóa toàn bộ index và các file liên quan.
    Dùng cho việc reset hoặc gỡ lỗi.
    """
    try:
        db_handler.clear_all()
        return {"message": "All data has been cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- Entry point để chạy ứng dụng ---
# Điều này cho phép chạy file bằng lệnh `python main.py`
if __name__ == "__main__":
    print(f"Starting server with the following settings:")
    print(f"  - Vector Dimension: {settings.VECTOR_DIMENSION}")
    print(f"  - Data Directory: {settings.DATA_DIR}")
    print(f"  - Listening on: {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=True
    )
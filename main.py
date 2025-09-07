# main.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import json

# Import class handler từ file faiss_handler.py
from faiss_handler import FaissHandler
from config import settings

# --- Pydantic Models để validate dữ liệu đầu vào ---

class VectorInput(BaseModel):
    product_id: str = Field(..., description="ID của sản phẩm, liên kết với CSDL quan hệ.")
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

@app.post("/upload_vectors", summary="Upload file JSON chứa nhiều vector sản phẩm")
async def upload_vectors(file: UploadFile = File(...)):
    """
    Upload một file JSON chứa danh sách các đối tượng { "product_id": int, "vector": List[float] }.
    Dữ liệu sẽ được thêm vào cơ sở dữ liệu FAISS.
    """
    if file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file JSON.")

    try:
        content = await file.read()
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="File không phải là JSON hợp lệ.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc file: {e}")

    if not isinstance(data, dict) or "vectors" not in data or not isinstance(data["vectors"], list):
        raise HTTPException(status_code=400, detail="Nội dung JSON phải là một đối tượng có khóa 'vectors' chứa danh sách các đối tượng.")

    vectors_data = data["vectors"]

    added_count = 0
    failed_count = 0
    errors = []

    for i, item_data in enumerate(vectors_data):
        try:
            # Map 'embedding' to 'vector' for Pydantic model
            item_data_mapped = {"product_id": item_data.get("product_id"), "vector": item_data.get("embedding")}
            item = VectorInput(**item_data_mapped)
            if len(item.vector) != db_handler.dimension:
                errors.append(f"Dòng {i+1} (product_id: {item.product_id}): Chiều vector không khớp. Mong đợi {db_handler.dimension}, nhận được {len(item.vector)}.")
                failed_count += 1
                continue

            db_handler.add(item.product_id, item.vector)
            added_count += 1
        except Exception as e:
            errors.append(f"Dòng {i+1}: Lỗi khi thêm dữ liệu (product_id: {item_data.get('product_id', 'N/A')}): {e}")
            failed_count += 1

    # Save the index after all vectors from the file have been processed
    db_handler.save()

    return {
        "message": "Hoàn tất xử lý file.",
        "total_records_in_file": len(vectors_data),
        "vectors_added_successfully": added_count,
        "vectors_failed_to_add": failed_count,
        "errors": errors if errors else None,
        "total_vectors_in_db": db_handler.get_total_vectors()
    }


    
# --- Entry point để chạy ứng dụng ---
# Điều này cho phép chạy file bằng lệnh `python main.py`
if __name__ == "__main__":
    print("Starting server with the following settings:")
    print(f"  - Vector Dimension: {settings.VECTOR_DIMENSION}")
    print(f"  - Data Directory: {settings.DATA_DIR}")
    print(f"  - Listening on: {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=True
    )
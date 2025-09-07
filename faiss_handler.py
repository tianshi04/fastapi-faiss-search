# faiss_handler.py

import faiss
import numpy as np
import os
import json
import threading
from typing import List, Tuple

from config import settings

class FaissHandler:
    def __init__(self):
        self.data_dir = settings.DATA_DIR
        self.index_path = os.path.join(self.data_dir, settings.INDEX_FILE)
        self.ids_path = os.path.join(self.data_dir, settings.IDS_FILE)
        
        self.dimension = settings.VECTOR_DIMENSION  # QUAN TRỌNG: Phải khớp với chiều vector của bạn
        self.index = None
        self.id_map: List[str] = []
        
        # Lock để đảm bảo việc ghi và thêm dữ liệu là thread-safe
        self.lock = threading.Lock()
        
        # Tạo thư mục data nếu chưa có
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.load()

    def load(self):
        """Tải index và map ID từ file vào bộ nhớ."""
        print("Loading FAISS index and ID map...")
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.ids_path, 'r') as f:
                    self.id_map = json.load(f)
                
                # Kiểm tra dimension
                if self.index.d != self.dimension:
                    raise ValueError(f"Dimension mismatch! Index has {self.index.d}, but config needs {self.dimension}.")
                
                print(f"Successfully loaded index with {self.index.ntotal} vectors.")
            else:
                print("No existing index found. Initializing a new one.")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.id_map: List[str] = []
        except Exception as e:
            print(f"Error loading data: {e}. Re-initializing.")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_map = []

    def save(self):
        """GHI ĐÈ TRỰC TIẾP lên file. Đơn giản nhưng có rủi ro hỏng file."""
        print("Saving data by overwriting existing files...")
        try:
            # Ghi đè trực tiếp
            faiss.write_index(self.index, self.index_path)
            with open(self.ids_path, 'w') as f:
                json.dump(self.id_map, f)
            print("Save complete.")
        except Exception as e:
            # Nếu lỗi xảy ra, có thể file đã bị hỏng
            print(f"CRITICAL: Error saving data, files might be corrupted! Error: {e}")

    def add(self, product_id: str, vector: List[float]):
        """Thêm một cặp (ID, vector) mới và lưu lại."""
        with self.lock:
            vector_np = np.array([vector]).astype('float32')
            self.index.add(vector_np)
            self.id_map.append(product_id)
            
            return self.index.ntotal

    def search(self, vector: List[float], k: int = 1) -> List[Tuple[str, float]]:
        """Tìm kiếm k hàng xóm gần nhất cho một vector."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.array([vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(min(k, len(indices[0]))):
            result_idx = indices[0][i]
            if result_idx == -1:
                continue

            found_id = self.id_map[result_idx]
            dist = distances[0][i]
            # Tính toán điểm tin cậy dựa trên khoảng cách.
            # Công thức này chuẩn hóa khoảng cách L2 (dist) thành một giá trị từ 0.0 đến 1.0.
            # dist = 0.0 (khoảng cách hoàn hảo) sẽ cho confidence = 1.0.
            # dist = settings.DISTANCE_NORMALIZATION_FACTOR sẽ cho confidence = 0.0.
            # Các khoảng cách lớn hơn NORMALIZATION_FACTOR sẽ bị cắt về 0.0.
            confidence = max(0.0, 1.0 - (dist / settings.DISTANCE_NORMALIZATION_FACTOR))
            results.append((found_id, float(confidence)))
            
        return results
    
    def get_total_vectors(self) -> int:
        """Trả về tổng số vector trong index."""
        return self.index.ntotal if self.index else 0

    def clear_all(self):
        """Xóa toàn bộ dữ liệu."""
        with self.lock:
            # Tạo lại index mới
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_map: List[str] = []
            
            # Gọi save để ghi đè các file cũ (hoặc tạo file rỗng)
            self.save()
            print("All data has been cleared.")
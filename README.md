# Dog Face Recognition API

A FastAPI-based dog face recognition system that uses YOLOv8 for detection, ArcFace for embedding generation, and FAISS for efficient similarity search.

## Features

- **Dog Detection**: Uses YOLOv8m.pt model to detect dogs in images
- **Face Embedding**: Generates 512-dimensional face embeddings using ArcFace (insightface)
- **Similarity Search**: Fast similarity search using FAISS (Facebook AI Similarity Search)
- **RESTful API**: Complete REST API with automatic documentation
- **Database Management**: Persistent storage of dog information and embeddings

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Upload Image  │───▶│  YOLOv8 Detect  │───▶│ Extract Face    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FAISS Search  │◀───│  Store in DB    │◀───│ ArcFace Embed   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- pip
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd scan-backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 model** (optional - will be downloaded automatically on first run)
   ```bash
   # The model will be downloaded automatically when the app starts
   # or you can manually download yolov8m.pt to the project root
   ```

## Usage

### Starting the Server

```bash
# Development mode
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the startup script
python start.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Register a Dog
```http
POST /dogs/register
Content-Type: multipart/form-data

Form Data:
- image: Dog image file (JPG, PNG, BMP)
- name: Dog's name (required)
- breed: Dog's breed (optional)
- owner: Owner's name (optional)
- description: Additional description (optional)
```

**Response:**
```json
{
  "success": true,
  "message": "Dog 'Buddy' registered successfully with 2 face embeddings",
  "dog_id": "uuid-string",
  "embedding_count": 2
}
```

#### 2. Recognize a Dog
```http
POST /dogs/recognize
Content-Type: multipart/form-data

Form Data:
- image: Dog image file (JPG, PNG, BMP)

Query Parameters:
- top_k: Number of top matches (default: 5, max: 20)
- threshold: Minimum similarity threshold (default: 0.6, range: 0.0-1.0)
```

**Response:**
```json
{
  "success": true,
  "message": "Found 3 similar dogs",
  "matches": [
    {
      "dog_id": "uuid-string",
      "dog_info": {
        "id": "uuid-string",
        "name": "Buddy",
        "breed": "Golden Retriever",
        "owner": "John Doe",
        "description": "Friendly dog",
        "created_at": "2024-01-01T12:00:00"
      },
      "similarity_score": 0.95,
      "distance": 0.05
    }
  ],
  "processing_time": 0.234
}
```

#### 3. Database Information
```http
GET /dogs/database/info
```

**Response:**
```json
{
  "total_dogs": 10,
  "total_embeddings": 25,
  "database_size_mb": 2.34
}
```

#### 4. List All Dogs
```http
GET /dogs/list
```

#### 5. Remove a Dog
```http
DELETE /dogs/{dog_id}
```

#### 6. Health Check
```http
GET /dogs/health
```

## Configuration

The application can be configured using environment variables or by modifying `app/core/config.py`:

```python
# Model settings
YOLO_MODEL_PATH=yolov8m.pt
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.4

# FAISS settings
EMBEDDING_DIMENSION=512  # ArcFace uses 512-dimensional embeddings
FAISS_INDEX_TYPE=Flat

# Storage settings
UPLOAD_DIR=uploads
EMBEDDINGS_DIR=embeddings
DATABASE_PATH=dog_database.faiss

# API settings
MAX_FILE_SIZE=10485760  # 10MB
```

## Project Structure

```
scan-backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── dogs.py          # API endpoints
│   ├── core/
│   │   └── config.py            # Configuration settings
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   ├── services/
│   │   ├── detection_service.py # YOLOv8 detection
│   │   ├── embedding_service.py # ArcFace embedding generation
│   │   └── faiss_service.py     # FAISS similarity search
│   └── main.py                  # FastAPI application
├── uploads/                     # Uploaded images
├── embeddings/                  # Stored embeddings
├── dog_database.faiss          # FAISS index
├── dog_database_metadata.json  # Database metadata
├── requirements.txt            # Python dependencies
├── start.py                    # Startup script
├── test_setup.py               # Setup verification script
└── README.md                   # This file
```

## Technical Details

### Detection Pipeline
1. **Image Upload**: Accepts various image formats
2. **YOLOv8 Detection**: Detects dogs using YOLOv8m.pt model
3. **Face Extraction**: Extracts face regions from detected dogs
4. **Preprocessing**: Resizes and normalizes face images to 112x112 RGB for ArcFace

### Embedding Generation
- Uses **ArcFace** (insightface library) for state-of-the-art face recognition
- Generates 512-dimensional embeddings (vs 128 for face_recognition)
- Better accuracy and robustness compared to traditional methods
- L2 normalization for consistent similarity computation

### Similarity Search
- FAISS IndexFlatIP for inner product similarity
- Cosine similarity between normalized embeddings
- Configurable threshold and top-k results

### Database Storage
- FAISS index for fast similarity search
- JSON metadata for dog information
- Persistent storage across application restarts

## Performance Considerations

- **Model Loading**: YOLOv8 and ArcFace models are loaded once at startup
- **Batch Processing**: Multiple faces can be processed in a single image
- **Memory Management**: Images are processed in memory and not stored permanently
- **FAISS Optimization**: Uses efficient similarity search algorithms
- **ArcFace Benefits**: Higher accuracy with 512-dimensional embeddings

## Troubleshooting

### Common Issues

1. **ArcFace Installation Issues**
   ```bash
   # Install insightface
   pip install insightface
   
   # If you encounter issues, try:
   pip install --upgrade pip
   pip install insightface --no-cache-dir
   ```

2. **CUDA Issues**
   - The current setup uses CPU-only FAISS and ArcFace
   - For GPU acceleration, install `faiss-gpu` and configure ArcFace for GPU

3. **Memory Issues**
   - Reduce `MAX_FILE_SIZE` in configuration
   - Process smaller images
   - Monitor FAISS index size (512D embeddings use more memory)

4. **Model Download Issues**
   - ArcFace models are downloaded automatically on first use
   - Check internet connection for model downloads
   - Models are cached locally for subsequent runs

### Logs
- Check console output for detailed error messages
- Model loading and service initialization logs are displayed at startup

## Testing

Run the setup verification script to check if everything is configured correctly:

```bash
python test_setup.py
```

This will test:
- Package imports
- Application module imports
- Configuration loading
- Directory creation
- ArcFace model loading

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [InsightFace ArcFace](https://github.com/deepinsight/insightface)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)

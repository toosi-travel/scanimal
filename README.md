# Dog Face Recognition API

A FastAPI-based dog face recognition system using YOLOv8 and PostgreSQL.

## Features

- **Dog Registration**: Register new dogs with face images
- **Dog Recognition**: Recognize dogs from uploaded images
- **Duplicate Detection**: Detect potential duplicate dogs
- **PostgreSQL Database**: Store dog information and metadata
- **RESTful API**: Clean API endpoints for all operations

## Prerequisites

- Python 3.8+
- PostgreSQL database
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd scan-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your PostgreSQL database in `app/core/config.py`:
```python
# PostgreSQL settings
postgresql_user: str = "your_username"
postgresql_pass: str = "your_password"
postgresql_host: str = "your_host"
postgresql_db: str = "your_database"
postgresql_port: int = 5432
```

## Database Setup

1. Initialize the database:
```bash
python init_database.py
```

2. Or run migrations with Alembic:
```bash
alembic upgrade head
```

## Running the Application

### Option 1: Direct Python execution
```bash
python run.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Option 3: Using the start script
```bash
python start.py
```

## API Endpoints

### Dogs
- `POST /dogs/register` - Register a new dog
- `POST /dogs/recognize` - Recognize a dog from image
- `GET /dogs/list` - List all registered dogs
- `GET /dogs/database/info` - Get database information
- `DELETE /dogs/{dog_id}` - Remove a dog
- `GET /dogs/health` - Health check

### Duplicate Detection
- `POST /duplicate-detection/check` - Check multiple images for duplicates and return best matches
- `POST /duplicate-detection/check-single` - Check single image for duplicates and return best match
- `GET /duplicate-detection/pending` - Get pending duplicates
- `POST /duplicate-detection/pending/{id}/approve` - Approve duplicate
- `POST /duplicate-detection/pending/{id}/reject` - Reject duplicate
- `GET /duplicate-detection/thresholds` - Get detection thresholds
- `GET /duplicate-detection/stats` - Get system statistics
- `GET /duplicate-detection/logs` - Get processing logs

## Multi-Image Duplicate Detection

The new `/duplicate-detection/check` endpoint allows you to process multiple images simultaneously and get the best match for each image without auto-approving or rejecting.

### Request Format
```http
POST /duplicate-detection/check
Content-Type: multipart/form-data

Form Data:
- images: Multiple image files (JPG, PNG, BMP)
```

### Response Format
```json
{
  "success": true,
  "message": "Processed 3 images. 2 successful matches, 1 failed.",
  "total_images": 3,
  "successful_matches": 2,
  "failed_images": 1,
  "results": [
    {
      "image_index": 0,
      "success": true,
      "message": "Best match found: Buddy (Score: 0.923)",
      "best_match": {
        "dog_id": "uuid-string",
        "name": "Buddy",
        "breed": "Golden Retriever",
        "owner": "John Doe",
        "similarity_score": 0.923
      },
      "processing_time": 0.156
    }
  ],
  "total_processing_time": 0.234
}
```

### Key Features
- **Multiple Images**: Process up to 10+ images in a single request
- **Best Match Only**: Returns the best matching dog for each image
- **No Auto-Decisions**: You control whether to register or reject based on scores
- **Detailed Results**: Each image gets individual processing results
- **Performance Metrics**: Processing time for each image and total

### Single Image Endpoint

The `/duplicate-detection/check-single` endpoint provides the same functionality for a single image:

#### Request Format
```http
POST /duplicate-detection/check-single
Content-Type: multipart/form-data

Form Data:
- image: Single image file (JPG, PNG, BMP)
```

#### Response Format
```json
{
  "success": true,
  "message": "Best match found: Buddy",
  "best_match": {
    "dog_id": "uuid-string",
    "name": "Buddy",
    "breed": "Golden Retriever",
    "owner": "John Doe",
    "similarity_score": 0.923,
    "confidence": 0.923
  },
  "processing_time": 0.156
}
```

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **API info**: http://127.0.0.1:8000/info

## Testing

Test the configuration:
```bash
python test_config.py
```

Test PostgreSQL connection:
```bash
python test_postgresql_integration.py
```

## Bulk Registration

For bulk importing multiple dog images, use the bulk registration script:

### Basic Usage
```bash
# Register all images in a folder
python bulk_register.py /path/to/dog/images

# Use structured naming (Owner_Breed_Name.jpg)
python bulk_register.py /path/to/images --naming structured

# Dry run to see what would be registered
python bulk_register.py /path/to/images --dry-run

# Custom API URL
python bulk_register.py /path/to/images --api-url http://localhost:8000

# Add delay between requests
python bulk_register.py /path/to/images --delay 2.0
```

### Naming Strategies

1. **Filename Strategy** (default):
   - `buddy.jpg` → Name: "Buddy"
   - `golden_retriever_max.jpg` → Name: "Golden Retriever Max"

2. **Structured Strategy**:
   - `John_GoldenRetriever_Buddy.jpg` → Owner: "John", Breed: "Golden Retriever", Name: "Buddy"
   - `Sarah_Labrador_Max.jpg` → Owner: "Sarah", Breed: "Labrador", Name: "Max"

### Features
- **Multiple Formats**: Supports JPG, PNG, BMP, TIFF
- **Progress Tracking**: Shows progress for each image
- **Error Handling**: Continues processing even if some images fail
- **Rate Limiting**: Configurable delay between API requests
- **Dry Run Mode**: Test without actually registering dogs
- **Detailed Reporting**: Summary with success rates and error details

### Example Output
```
🚀 Starting bulk registration from: /path/to/dog/images
📋 Naming strategy: filename
🔍 Dry run: No
⏱️  Delay between requests: 1.0s
============================================================

📸 Processing image 1/3: buddy.jpg
   🐕 Name: Buddy
   📤 Registering...
   ✅ Successfully registered!
      🆔 Dog ID: 123e4567-e89b-12d3-a456-426614174000
      🔢 Embeddings: 2
   ⏳ Waiting 1.0s before next request...

📊 BULK REGISTRATION SUMMARY
============================================================
📁 Total images found: 3
✅ Successful registrations: 3
❌ Failed registrations: 0
🔍 Skipped images (dry run): 0

🎯 Success rate: 100.0%
🎉 Bulk registration completed successfully!
```

## Project Structure

```
app/
├── api/           # API routes and endpoints
├── core/          # Configuration and database setup
├── models/        # Database models
├── repositories/  # Data access layer
├── services/      # Business logic services
└── main.py        # FastAPI application entry point
```

## Configuration

The application uses Pydantic settings for configuration. Key settings include:

- **Model settings**: YOLO model path, confidence thresholds
- **Database settings**: PostgreSQL connection parameters
- **Storage settings**: Upload directories and file paths
- **API settings**: File size limits and allowed extensions

## Troubleshooting

### Socket Connection Issues
If you encounter socket connection errors:
1. Use `127.0.0.1` instead of `0.0.0.0` for the host
2. Disable reload mode by setting `reload=False`
3. Check your firewall and network settings

### Database Connection Issues
1. Verify PostgreSQL is running
2. Check connection parameters in config
3. Ensure database exists and user has proper permissions
4. Test connection with `test_postgresql_integration.py`

## License

[Your License Here]

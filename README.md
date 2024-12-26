
# 🤖 MLServiceCloud-Backend

Welcome to the backend repository of our Weather Image Classification project! This service provides a robust API for processing and classifying meteorological images using a convolutional neural network.

## 🎯 Project Overview

This backend service is built with Python and FastAPI to provide high-performance image classification capabilities. It's designed to handle large-scale image processing with efficient batch processing and memory management.

### 🚀 Key Features

- Fast and efficient image processing
- RESTful API endpoints
- Convolutional Neural Network implementation
- Batch processing support
- Scalable architecture
- Docker containerization

## 🛠️ Technologies Used

- **FastAPI** - Web Framework
- **Python 3.9.0** - Programming Language
- **TensorFlow** - Machine Learning Framework
- **Docker** - Containerization
- **uvicorn** - ASGI Server

## 📦 Dependencies

Key libraries used in this project:
```python
fastapi
tensorflow
numpy
python-multipart
uvicorn
```

## 🔧 Setup & Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MLServiceCloud-Backend.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## 🐳 Docker Support

Run the service using Docker:

```bash
# Build the image
docker build -t mlservice-backend .

# Run the container
docker run -p 8000:8000 mlservice-backend
```

## 🔌 API Endpoints

- `POST /api/predict`
  - Upload and classify weather images
  - Accepts image files in common formats (jpg, png)
  - Returns classification results and confidence scores

## 🎥 Demo Video

Check out our project demonstration video [here](video-link)! The video showcases the full functionality of our service and its integration with the frontend.

## 📝 Documentation

The complete project documentation and detailed report can be found in:
- [Project Report (PDF)](https://docs.google.com/document/d/10U-kktKWC5EB4KLxGytVVu6gyYVuLawf7wcvzxDWSBo/edit?usp=sharing)
- [Frontend Repository](https://github.com/JohnMarulanda/MLServiceCloud-Frontend)

## 🔬 Model Architecture

Our CNN model architecture includes:
- Multiple convolutional layers
- Max pooling layers
- Dropout for regularization
- Dense layers for classification
- Softmax activation for multi-class prediction

## 👥 Contributors

- [**Arango Guzman Juan Felipe**](https://github.com/JuanArango30)
- [**Guerrero Jaramillo Carlos Eduardo**](https://github.com/ClusterMax)
- [**Marulanda Valero John Jader**](https://github.com/JohnMarulanda)
- [**Rivera Reyes Miguel Angel**](https://github.com/BitzKort)

---
Made with 🧠 for Universidad del Valle - Neural Networks Course

# Indian Flower Classifier (TensorFlow + Streamlit)

A web app that classifies 10 common Indian flowers using a CNN trained with TensorFlow and served via Streamlit.

## 🏷️ Classes
- Marigold
- Hibiscus
- Rose
- Tecoma
- Jasmine
- Lotus
- Periwinkle
- Sunflower
- Bougainvillea
- Canna Lily

## 🔧 Setup & Run

### 1. Organize Dataset (Optional)
```bash
python organize_dataset.py
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run Streamlit App
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deployment
You can upload this to Streamlit Cloud or host locally. Make sure to include:
- `flower_model.h5`
- `class_names.json`
- All files in `streamlit_app/`

Happy Flower Classifying! 🌸
"# Flowers" 

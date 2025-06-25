# 🧠 ClassiFyAI – Smart Product Classifier

ClassiFyAI is an AI-powered product classification system that automatically categorizes products based on their name, description, and price. Designed for e-commerce and inventory systems, it uses machine learning to streamline and automate the classification process.

---

## 🚀 Features

- ✅ AI-based product category prediction
- 📄 Input product name, description, and price
- 🧠 Trained with XGBoost + TF-IDF pipeline
- 🔒 Secure user authentication with JWT & bcrypt
- 📦 RESTful API for managing product data
- 🎯 Clean and responsive UI using React + Tailwind CSS

---

## 🛠️ Tech Stack

| Layer      | Technologies |
|------------|--------------|
| Frontend   | React, Tailwind CSS, Vite |
| Backend    | FastAPI, Python, Scikit-learn, XGBoost |
| Database   | MySQL (via GORM in .NET backend) |
| Auth       | JWT, Bcrypt |
| DevOps     | Docker (for containerization) |

---

## 📂 Project Structure
/SmartProductClassifier
├── frontend/ # React + Tailwind app
├── backend/ # FastAPI + ML model
├── model/ # Trained model (XGBoost)
├── api/ # .NET API (for DB and auth)
└── data/ # Product dataset

### 1. Clone the Repository
git clone https://github.com/your-username/ClassiFyAI.git
cd ClassiFyAI
###2. Install Backend Dependencies (Python)
cd backend
pip install -r requirements.txt

##3. Start Backend (FastAPI)
uvicorn main:app --reload

4. Install Frontend Dependencies (React)
cd ../frontend
npm install
npm run dev

##📈 Model Training (Optional)
The model is trained using TF-IDF vectorization on text fields and XGBoost for classification. You can retrain the model by running:

python train_model.py

##🧪 Example Input
{
  "name": "Pepsi 500ml",
  "description": "Carbonated soft drink bottle",
  "price": 40
}

##Prediction Output:
"category": "Beverages"

##SCREENSHOTS:
![image](https://github.com/user-attachments/assets/3a9c297b-7fb1-4d4a-9ce2-d4e5b6571676)
![image](https://github.com/user-attachments/assets/8192005b-e950-45f3-b736-5e35ed916901)
![image](https://github.com/user-attachments/assets/440dcb06-a1c8-4bb8-a1e8-014bb2cda8da)
![image](https://github.com/user-attachments/assets/362eab30-03b7-4f15-bd1a-af12554de382)



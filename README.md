# 🦙 LLaMA Chatbot (Streamlit + Ollama)

A lightweight, local AI chatbot built using **Streamlit** and **Ollama**, powered by the **LLaMA 3.2** model.  
This project demonstrates how to run a modern large language model locally with an interactive web interface, without relying on any cloud services or external APIs.

---

## 🚀 Features

- Local LLaMA 3.2 inference via Ollama  
- Interactive chat interface using Streamlit  
- Session-based chat history  
- Simple setup and minimal dependencies  
- No API keys or cloud services required  

---

## 🛠 Tech Stack

- **Python 3.9+**
- **Streamlit**
- **Ollama**
- **LLaMA 3.2**
- **Requests**

---

## 📂 Project Structure

ollama-llama-streamlit-chatbot/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


---

## ⚙️ Prerequisites

Before running the application, ensure the following are installed:

- Python 3.9 or higher  
- Ollama (installed and running locally)  
- LLaMA 3.2 model available in Ollama  

---

## 🔧 Installation & Usage

### 1️⃣ Install Python Dependencies
  ```pip install -r requirements.txt```
### 2️⃣ Pull the LLaMA Model (if not already available)
  ```ollama run llama3.2```
### 3️⃣ Run the Application
  ```streamlit run app.py```

Once the app starts, open your browser and navigate to:
    ```http://localhost:8501```

🧠 How It Works
- Streamlit manages the user interface and session state.
- User prompts are sent to Ollama via its local REST API.
- The LLaMA 3.2 model processes prompts and generates responses locally.
- Chat history is preserved for the duration of the Streamlit session.

⚠️ Notes & Limitations
- Ollama must be running on http://localhost:11434
- Chat history is stored only in session memory and resets on page reload
- This project is intended for local experimentation and learning

- This project is intended for local experimentation and learning

📜 License
This project is licensed under the MIT License.

## 👤 Author

**Ravi Shankar Kumar**  
Aspiring Data & Machine Learning professional with a strong interest in building practical AI applications and local LLM-based systems.

- LinkedIn: [linkedin.com/in/ravi0901](https://www.linkedin.com/in/ravi0901/)


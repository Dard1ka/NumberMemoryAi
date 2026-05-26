# NumberMemoryAi
Number Memory AI is an Artificial Intelligence-based game that trains number memory and English pronunciation skills using Speech-to-Text technology.

Players will memorize a series of numbers (0–100), then recite them in order. The AI system will process the voice in real-time and provide accurate assessments based on a model that has been fine-tuned specifically for number recognition.

Objectives
- Train memory retention through gamification
- Improve pronunciation of numbers in English
- Explore the implementation of AI Speech Recognition in web applications

Model Files : https://drive.google.com/drive/folders/1jkbjfDwRxcyAHhWbErP5BsDfnzcyD2NU?usp=sharing 

## Demo Video in Indonesian : https://youtube.com/shorts/Lww2MENCgiA?si=LA4yWEaAAxXtHUDv

📂 File Structure
```pqsql
NumberMemoryAi/
│
├── app.py
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── model_output/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── processor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
```

Tech Stack
- Python
- FastAPI
- HTML / CSS / JavaScript
- Browser MediaRecorder API
- Safetensors

💻 Installation & Run

1️⃣ Clone Repository
```bash
git clone https://github.com/Dard1ka/NumberMemoryAi.git
cd NumberMemoryAi
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run Application
```bash
python app.py
```

4️⃣ Open in Browser
Open HTML in Live Server

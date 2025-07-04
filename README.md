# debug-assignment
This project is a FastAPI-based health assistant that processes blood test reports (PDFs), extracts key values, and provides:

🩸 Medical interpretations from an AI-powered doctor

🥦 Nutrition advice for deficiencies

🏋️ Personalized exercise plans

✅ Document verification

It uses Azure OpenAI, LangChain, and CrewAI to simulate a team of intelligent agents offering actionable health suggestions.

 Features
📂 Upload PDF Blood Reports

🤖 Multi-agent system: Includes a doctor, nutritionist, fitness expert, and verifier

🔍 Context-aware LLM reasoning over PDF content

🔒 Clean API with error handling and temporary file storage

🌐 Built with FastAPI, integrated with Azure OpenAI

Project Structure:
├── research.py          # Main FastAPI + Agent logic
├── requirements.txt     # Dependencies
└── data/                # Uploaded reports (temporary)

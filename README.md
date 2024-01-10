# llm-demo
Sample application to demonstrate langchain and ingesting custom docments for use in relevance search against OpenAI ChatGPT 3-5 Turbo.

Tested with Python 3.11 and at the time of writing 3.12 which did NOT work due to pytorch pip dependency not existing.

## Setup
- Register an OpenAI account and create an API key. Make sure it has funds as this demo will cost to run (not much, but still requires credits for the API key).

- Copy the environment example file:
```
cp .env.example .env
```

- Replace OPENAI_API_KEY in .env
- Done!


## Installing windows
Instructions are for Windows.
Run following commands to install dependencies into a virtual Python environment.
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

# Running
Run following commands to run the program with dependencies installed into the virtual environment:
```
.venv\Scripts\Activate.ps1
python main.py
```
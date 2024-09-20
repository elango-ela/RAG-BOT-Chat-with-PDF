Step 1: Open the Docker desktop

Step 2: Enter this command in cmd to run this RAGBOT space locally(Please replace the YOUR_VALUE_HERE with your google api key to run this space locally)


docker run -it -p 7860:7860 --platform=linux/amd64 \ -e api_key="YOUR_VALUE_HERE" \ registry.hf.space/inela-ragbot:latest python app.py

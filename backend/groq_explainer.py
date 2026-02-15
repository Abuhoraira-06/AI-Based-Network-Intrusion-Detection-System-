import os
import httpx

def groq_explain(sample, prediction, confidence):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Groq API key not configured."

    port, flow, packets, pkt_len, active = sample[0]
    label = "Malicious" if prediction == 1 else "Benign"

    prompt = f"""
You are a cybersecurity analyst.

Traffic details:
Destination Port: {port}
Flow Duration: {flow}
Total Forward Packets: {packets}
Packet Length Mean: {pkt_len}
Active Mean: {active}

Prediction: {label}
Confidence: {confidence}%

Explain briefly why this traffic is classified as {label}.
"""

    try:
        response = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq API Error: {response.text}"

    except Exception as e:
        return f"Connection Error: {str(e)}"

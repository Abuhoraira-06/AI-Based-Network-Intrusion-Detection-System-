import os

def groq_explain(sample, prediction, confidence):
    try:
        from groq import Groq
    except ImportError:
        return "Groq module not available in deployment environment."

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "Groq API key not configured."

    try:
        client = Groq(api_key=api_key)

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

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"

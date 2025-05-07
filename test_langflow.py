import requests
import json
import sys

# Set UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# === Langflow API Configuration ===
LANGFLOW_URL = st.secrets["LANGFLOW_URL"]
LANGFLOW_TOKEN = st.secrets["LANGFLOW_KEY"]

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        # Fall back to ASCII if UTF-8 fails
        print(text.encode('ascii', 'replace').decode())

def test_langflow():
    safe_print("Sending request to Langflow...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LANGFLOW_TOKEN}",
    }
    
    payload = {
        "input_value": "I'm feeling anxious about my future.",
        "input_type": "chat",
        "output_type": "chat",
    }
    
    try:
        safe_print(f"Making request to URL: {LANGFLOW_URL}")
        safe_print(f"Headers: {json.dumps(headers, indent=2)}")
        safe_print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(LANGFLOW_URL, json=payload, headers=headers, timeout=30)
        safe_print(f"\nResponse Status: {response.status_code}")
        safe_print(f"\nResponse Headers: {json.dumps(dict(response.headers), indent=2)}")
        
        # Try to get the raw response text first
        raw_text = response.text
        safe_print("\nRaw Response Text:")
        safe_print(raw_text)
        
        # Try to parse the JSON
        try:
            json_response = response.json()
            safe_print("\nParsed JSON:")
            safe_print(json.dumps(json_response, indent=2))
            
            # Extract the actual message
            if 'outputs' in json_response and len(json_response['outputs']) > 0:
                output = json_response['outputs'][0]
                if 'outputs' in output and len(output['outputs']) > 0:
                    result = output['outputs'][0]
                    if 'results' in result and 'message' in result['results']:
                        message = result['results']['message']
                        if 'text' in message:
                            safe_print("\nExtracted Message:")
                            safe_print(message['text'])
                        elif 'data' in message and 'text' in message['data']:
                            safe_print("\nExtracted Message:")
                            safe_print(message['data']['text'])
        except json.JSONDecodeError as e:
            safe_print(f"\nError parsing JSON: {str(e)}")
            safe_print("Raw response text was not valid JSON")
            
    except requests.exceptions.RequestException as e:
        safe_print(f"Error making request: {str(e)}")

if __name__ == "__main__":
    test_langflow() 
import requests
import json
import time

print('Starting API request...')
with open('20260422101955-1776853170.13-02712471468-1036-Inbound.wav', 'rb') as f:
    files = {'file': f}
    data = {'language': 'vi', 'diarize': 'true'}
    # This is SSE stream, so we just iterate lines
    response = requests.post('http://127.0.0.1:8000/api/transcribe-stream', files=files, data=data, stream=True)
    
    with open('test_output.txt', 'w', encoding='utf-8') as out_f:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    event = json.loads(line_str[6:])
                    if event['type'] == 'segment':
                        msg = f"[{event['start']} -> {event['end']}] {event.get('speaker')}: {event['text']}\n"
                        out_f.write(msg)
                        out_f.flush()
                        # Chỉ print không dấu ra console để debug an toàn
                        print(f"Segment received: {event['start']} -> {event['end']}")
                    elif event['type'] == 'done':
                        print('Done!')
                        break



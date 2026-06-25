# This file includes classe to get response from deployed local LLM
import json
import os
import threading
import requests


class InterfaceLocalLLM:
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, url, timeout=None):
        urls = [item.strip() for item in str(url).split(",") if item.strip()]
        self._urls = urls or [url]
        self._url = self._urls[0]  # 'http://127.0.0.1:11045/completions'
        self._timeout = float(timeout) if timeout is not None else 180.0
        self._counter = 0

    def get_response(self, content: str) -> str:
        while True:
            try:
                response = self._do_request(content)
                return response
            except:
                continue

    def _do_request(self, content: str) -> str:
        content = content.strip('\n').strip()
        url = self._next_url()
        if "/v1/completions" in url:
            data = {
                'model': 'local',
                'prompt': content,
                'max_tokens': 768,
                'temperature': 1.0,
                'top_p': 0.98,
                'seed': 2024,
            }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, data=json.dumps(data), headers=headers, timeout=self._timeout)
            print(response)
            if response.status_code == 200:
                return response.json()['choices'][0]['text']
            return None

        # repeat the prompt for batch inference (inorder to decease the sample delay)
        data = {
            'prompt': content,
            'repeat_prompt': 1,
            'params': {
                'do_sample': True,
                'temperature': 1.0,
                'top_k': None,
                'top_p': 0.98,
                'add_special_tokens': False,
                'skip_special_tokens': True,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=self._timeout)
        print(response)
        if response.status_code == 200:
            response = response.json()['content'][0]
            return response

    def _next_url(self):
        if len(self._urls) == 1:
            return self._urls[0]
        self._counter += 1
        idx = (os.getpid() + threading.get_ident() + self._counter) % len(self._urls)
        return self._urls[idx]

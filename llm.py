import os
import requests
from pydantic import Field
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.language_models.llms import LLM

class OllamaLlama3(LLM):

    host: str = Field(None, alias='service_host')
    port: int = Field(None, alias='service_port')
    path: str = Field(None, alias='service_path')
    stop_words = Field(None, alias='stop_words')

    def __init__(self, host, port, path='api/generate'):
        super(OllamaLlama3, self).__init__()
        self.host = host
        self.port = port
        self.path = path

        self.stop_words = [
                    '<|start_header_id|>',
                    '<|end_header_id|>',
                    '<|eot_id|>'
                ]

    @property
    def _get_model_default_parameters(self):
        return {
            'model_name': 'Ollama Llama3',
            'model_port': 11434,
            'model_path': 'api/generate',
            'stop_words': [ '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>' ]
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': 'Ollama-LLM Llama3',
            'model_host': self.host,
            'model_port': self.port,
            'model_path': self.path
        }

    @property
    def _llm_type(self) -> str:
        return 'TensorRT-LLM Llama3'


    def _call(self, prompt: str, stop: Optional[List[str]]=None, **kwargs):
        target_url = f'http://{self.host}:{self.port}/{self.path}'

        stop_words = self.stop_words[:]
        if stop:
            stop_words.extend(stop)


        payload ={
                'model': 'llama3:instruct',
                'prompt': prompt,
                'stream': False,
                'stop': stop_words
        }

        resp = requests.post(target_url, json=payload)

        if resp.ok:
            return resp.json()['response']

        print(f'fail: {resp.json()}')
        return None

"""
    refer to https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/
    encapsulate the TensorRT version Llama3 with langchain LLM interface
    seems a simple REST api wrapper
"""
class TensorRTLlama3(LLM):

    host: str = Field(None, alias='service_host')
    port: int = Field(None, alias='service_port')
    path: str = Field(None, alias='service_path')
    stop_words = Field(None, alias='stop_words')

    def __init__(self, host, port, path='v2/models/ensemble/generate'):
        super(TensorRTLlama3, self).__init__()
        self.host = host
        self.port = port
        self.path = path

        self.stop_words = [
                    '<|start_header_id|>',
                    '<|end_header_id|>',
                    '<|eot_id|>'
                ]

    @property
    def _get_model_default_parameters(self):
        return {
            'model_name': 'TensorRT-LLM Llama3',
            'model_port': 8000,
            'model_path': 'v2/models/ensemble/generate',
            'stop_words': [ '<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>' ]
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': 'TensorRT-LLM Llama3',
            'model_host': self.host,
            'model_port': self.port,
            'model_path': self.path
        }

    @property
    def _llm_type(self) -> str:
        return 'TensorRT-LLM Llama3'


    def _call(self, prompt: str, stop: Optional[List[str]]=None, **kwargs):
        target_url = f'http://{self.host}:{self.port}/{self.path}'

        stop_words = self.stop_words[:]
        if stop:
            stop_words.extend(stop)

        payload ={
                'text_input': prompt,
                'max_tokens': kwargs['max_tokens'] if 'max_tokens' in kwargs else 100,
                'stop_words': stop_words
        }

        resp = requests.post(target_url, json=payload)

        if resp.ok:
            return resp.json()['text_output']

        print(f'fail: {resp.json()}')
        return None


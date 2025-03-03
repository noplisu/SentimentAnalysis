'''This module uses IBM Watson to return a sentiment analysis for a string'''
import json
import requests

def sentiment_analyzer(text_to_analyse):
    '''Analyze sentiment and return an object with label and score'''

    root = 'https://sn-watson-sentiment-bert.labs.skills.network'
    url = f'{root}/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'

    myobj = { "raw_document": { "text": text_to_analyse } }

    header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}

    response = requests.post(url, json=myobj, headers=header, timeout=5000)

    formatted_response = json.loads(response.text)

    label = None
    score = None
    if response.status_code == 200:
        label = formatted_response['documentSentiment']['label']
        score = formatted_response['documentSentiment']['score']

    return {'label': label, 'score': score}

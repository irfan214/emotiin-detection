
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EmotionOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

def emotion_predictor(text):
    authenticator = IAMAuthenticator('your_ibm_watson_api_key')
    
    nlu = NaturalLanguageUnderstandingV1(
        version='2021-08-01',
        authenticator=authenticator
    )
    
    nlu.set_service_url('your_service_url')

    response = nlu.analyze(
        text=text,
        features=Features(emotion=EmotionOptions())
    ).get_result()

    emotions = response['emotion']['document']['emotion']
    return {
        'anger': emotions['anger'],
        'fear': emotions['fear'],
        'joy': emotions['joy'],
        'sadness': emotions['sadness'],
        'disgust': emotions['disgust']
    }

# Test the function (optional)
# print(emotion_predictor("I am so happy today!"))

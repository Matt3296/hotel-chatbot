import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from enum import Enum

### Loading the Huggingface Token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key= hf_token,
)

### Next step: make chatbot class and have instances - e.g. sentiment analysis then bert gets selected
class Models(Enum):
    LLAMA: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    BERT: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    BLOSSOM: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    GGL: str = "google-t5/t5-base"
    TAB: str = "tabularisai/multilingual-sentiment-analysis"

MODEL_DESC = {
    "en": {
        Models.LLAMA: "Excellent for sentiment analysis.",
        Models.BERT: "Excellent for topic classification.",
        Models.BLOSSOM: "Excellent choice for Korean language tasks like conversation, Q&A, and summarization.",
        Models.GGL: "Excellent general purpose model. However, it struggles with non-Latin scripts (Cyrillic, Korean, Arabic, etc.)",
        Models.TAB: "Excellent choice for social media analysis, customer feedback analysis, market research, and customer service optimization."
    },
    
    "kr": {
        Models.LLAMA: "감정 분석에 탁월한 선택입니다.",
        Models.BERT: "토픽 분류에 탁월한 선택입니다.",
        Models.BLOSSOM: "대화, Q&A, 요약 등 한국어 업무에 탁월한 선택.",
        Models.GGL: "우수한 범용 모델. 그러나 비라틴 문자(키릴 문자, 한국어, 아랍어 등)에는 어려움이 있습니다.",
        Models.TAB: "소셜 미디어 분석, 고객 피드백 분석, 시장 조사 및 고객 서비스 최적화를 위한 탁월한 선택."
    }
}

def recommend_eng(lang: str = "en") -> None:
    for model in Models:
        try:
            description = MODEL_DESC[lang].get(model, "No description available.")
            print(f"{model.name}: {description}")
        except ValueError:
            print("Invalid model. Please pick one from the list.")

def recommend_kr(lang: str = "kr") -> None:
    for model in Models:
        try:
            description = MODEL_DESC[lang].get(model, "설명이 없습니다.")
            print(f"{model.name}: {description}")
        except ValueError:
            print("모델이 잘못되었습니다. 목록에서 하나를 선택해 주세요.")

class Chatbot:
    def __init__(self, model: Models):
        self.model = model

    # Review response
    def review_response(self) -> str:
        text = input("Enter text: ")

        completion = client.chat.completions.create(
            model=self.model.value,
            messages=[
                {
                    "role": "user",
                    "content": f"Respond to this text: {text}"
                }
            ],
        )

        # print(completion.choices[0].message)
        response = completion.choices[0].message
        print(response)
        return response

    # Prompt response
    def prompt_response(self) -> str:
        text = input("Enter text: ")

        completion = client.chat.completions.create(
            model=self.model.value,
            messages=[
                {
                    "role": "user",
                    "content": f"Respond to this prompt: {text}"
                }
            ],
        )

        response = completion.choices[0].message
        print(response)
        return response
    
    # Sentiment Analysis
    def sentiment_score(self) -> str:
        text = input("Enter text: ")

        result = client.text_classification(
        text,
        model=self.model.value,
        )

        print(result)
        return result
    
    # Translation
    def translation(self) -> str:
        src = input("Which language are you translating from? ")
        tgt = input("Which language would you like to translate to? ")

        text = input("Enter text: ")

        result = client.translation(
        text,
        model=self.model.value,
        src_lang=src,
        tgt_lang=tgt
        )

        # print(result)
        return result.translation_text

### Korean version
class Chatbot_kr(Chatbot):
    '''
    Translates detected language to Korean and then responds in the detected language.
    '''
    def __init__(self, model: Models):
        super().__init__(model)
        
    def sentiment_kr(self) -> str:
        text = input("텍스트를 입력하세요: ")

        result = client.text_classification(
        text,
        model=self.model.value,
        )

        print(result)
        return result

    def review_response_kr(self) -> str:
        text = input("텍스트를 입력하세요: ")

        completion = client.chat.completions.create(
            model=self.model.value,
            messages=[
                {
                    "role": "user",
                    "content": f"Respond to this text: {text}"
                }
            ],
        )

        response = completion.choices[0].message
        print(response)
        return response

    def translation_kr(self) -> str:
        '''
        Only translates to Korean.
        '''
        src = input("어떤 언어에서 번역하고 계신가요? ")

        text = input("텍스트를 입력하세요: ")

        result = client.translation(
        text,
        model=self.model.value,
        src_lang=src,
        tgt_lang="Korean"
        )

        # print(result)
        return result.translation_text

# recommend()
# kr_bot = chatbot_kr(model=Models.TAB)
# kr_bot.sentiment_kr()
bot = Chatbot(model=Models.GGL)
bot.translation()
# recommend_eng()


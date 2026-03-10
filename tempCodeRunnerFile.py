client = InferenceClient(
#     provider="hf-inference",
#     api_key=os.environ["HF_TOKEN"],
# )

# def sentiment_score():
#     text = input("Enter text: ")

#     result = client.text_classification(
#     text,
#     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
#     )

#     print(result)
#     return result

# sentiment_score()
from flask import Flask, request, render_template
from transformers import BertTokenizer
from model import Classifier
import torch

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = Classifier()
model_path = 'model_save/model1'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def sentiment_analysis(text, model, tokenizer):
    inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    prediction = torch.argmax(outputs, dim=1)

    return 'Positive' if prediction == 1 else 'Negative'

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        text = request.form.get("text")
        sentiment = sentiment_analysis(text, model, tokenizer)
        return render_template("index.html", sentiment=sentiment, text=text)
    return render_template("index.html", sentiment="", text="")

def main():
    print("這是一個能夠分析你言論的情緒是正面還是負面的語言模型（輸入'quit'後即可退出）：")
    while True:
        text = input("請輸入你的言論：")
        if text == 'quit':
            break
        sentiment = sentiment_analysis(text, model, tokenizer)
        print(f"情感傾向：{sentiment}")

if __name__ == "__main__":
    app.run(debug=True)
    #main()

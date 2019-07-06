from bert_serving.client import BertClient
import pandas
import numpy
import pickle

def load_model():
    with open("news_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    df = pandas.read_csv("data/sample.csv").head(20)
    with BertClient(ip="0.0.0.0") as client:
        vecs = client.encode(list(df["News"]))
    pred = pandas.DataFrame(vecs)
    model = load_model()
    label = model.predict(numpy.array(pred))
    df["pred_label"] = label
    df.query("pred_label == 1")[["News"]].to_csv("squeezed_news.csv", index=False)
    df.query("pred_label == 0")[["News"]].to_csv("gomi_news.csv", index=False)

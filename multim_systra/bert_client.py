from bert_serving.client import BertClient

if __name__ == "__main__":
    text_list = [
        "China",
        "Hello, world"
    ]

    with BertClient(ip="0.0.0.0") as client:
        vec = client.encode(text_list)
    print(vec)

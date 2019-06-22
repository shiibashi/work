from bert_serving.client import BertClient
import sentencepiece

if __name__ == "__main__":
    text_list = [
        "China",
        "Hello, world"
    ]

    with BertClient(ip="0.0.0.0") as client:
        vec = client.encode(text_list)
    print(vec)

"""
from bert_serving.client import BertClient
bc = BertClient(ip='0.0.0.0')

import sentencepiece as spm
s = spm.SentencePieceProcessor()
s.Load('./bert-jp/wiki-ja.model')

def parse(text):
    text = text.lower()
    return s.EncodeAsPieces(text)

texts = ['液体の水は、惑星上における生命の前提条件である。',
        '火星は地球型惑星に分類される。']

parsed_texts = list(map(parse,texts))

print(bc.encode(parsed_texts,is_tokenized=True))
"""
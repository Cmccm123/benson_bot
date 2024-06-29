from transformers import AutoModel
from numpy.linalg import norm
import chromadb
import json
import re
chroma_client = chromadb.PersistentClient(path="benson")
try:
    chroma_client.delete_collection("benson")
except:
    pass
cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True) # trust_remote_code is needed to use the encode method

def Find(string):
 
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


with open('data/benson_chat.json', 'r', encoding='utf-8') as json_file:
    odata = json.load(json_file)
data = []
for i in odata:
    if(len(Find(i)) == 0):
        data.append(i)
new_data=[]
for i in range(0,len(data),5):
    word = ""
    for j in data[i:i+10]:
        word+=j+"\n\n\n"
    new_data.append(word)
data=new_data
embeddings = model.encode(data)
    
collection = chroma_client.create_collection(name="benson")
collection.add(
    embeddings=embeddings.tolist(),
    documents=data,
    ids=[f"id{i}" for i in range(0,len(data))]
)
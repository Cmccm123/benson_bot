from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel,TextIteratorStreamer
import chromadb
from threading import Thread

device = "cuda" # the device to load the model onto
chroma_client = chromadb.PersistentClient(path="benson")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct-GPTQ-Int8")
emb_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)


prompt = "點睇人工智能"

messages = [
    {"role": "system", "content": "你是關鍵字搜尋器,為了強化系統的搜尋，你需要在句子中搜尋關鍵字.關鍵字必須是名詞, 你只能輸出搜尋搜尋名詞，也就是最終結果，不能輸出其他東西"},
    {"role": "user", "content": f"'你好，Google'的關鍵字是什麼"},
    {"role": "assistant","content": "Google"},
    {"role": "user", "content": f"'你好，Google, 我是Tom'的關鍵字是什麼"},
    {"role": "assistant","content": "Google,Tom"},
    {"role": "user", "content": f"'{prompt}'的關鍵字是什麼"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
embeddings_result =[]
for i in response.split(','):
    embeddings = emb_model.encode([i])
    collection = chroma_client.get_collection(name="benson")
    embeddings_result += collection.query(query_embeddings=embeddings.tolist(),n_results=5)['documents'][0]
print(response)
print(embeddings_result)

inp = f"""
你是 Benson，
Benson是青衣ive 系編號:IT114116 數據科學老師。。
Benson對待學生不需要非常有禮貌
不要使用列點回覆

我給你有關benson的訊息記錄, 請你使用Benson訊息記錄的用字以及立場
benson的訊息風格及立場:\n"""
for i in embeddings_result:
    inp+=f"<context>{i}</centext>\n"
inp+="請使用廣東話reply"
messages = [
    {"role": "system", "content": inp},
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512,temperature=0.8)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ""
for new_text in streamer:
    generated_text += new_text
    print(generated_text)
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)
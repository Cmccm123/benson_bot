import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, TextIteratorStreamer
import chromadb
import heapq
from threading import Thread
from FlagEmbedding import FlagReranker
from operator import itemgetter
device = "cuda" # the device to load the model onto
chroma_client = chromadb.PersistentClient(path="benson")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) 

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
emb_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)

def generate_response(prompt):
    key_word_messages = [
        {"role": "system", "content": "你是關鍵字搜尋器,你需要為一句句子創造關鍵字,關鍵字並不一定是句子中的詞語,而是要最能夠幫助系統在數據庫中搜尋到有用資料的關鍵字,\n你只能輸出搜尋搜尋名詞，也就是最終結果，不能輸出其他東西"},
        {"role": "user", "content": f"'你好，Google'的關鍵字是什麼"},
        {"role": "assistant", "content": "Google"},
        {"role": "user", "content": f"'太郎你鍾意食咩"},
        {"role": "assistant", "content": "太郎食"},
        {"role": "user", "content": f"'什麼是宇宙大爆炸"},
        {"role": "assistant", "content": "宇宙大爆炸"},
        {"role": "user", "content": f"'對於香港it的生態環境有什麼看法"},
        {"role": "assistant", "content": "香港it的看法"},
        {"role": "user", "content": f"'{prompt}'的關鍵字是什麼"}
    ]
    
    text = tokenizer.apply_chat_template(
        key_word_messages,
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
    print(response)
    embeddings_result =[]
    embeddings = emb_model.encode([str(response)])
    collection = chroma_client.get_collection(name="benson")
    embeddings_result += collection.query(query_embeddings=embeddings.tolist(),n_results=5)['documents'][0]
    embeddings_result_clean = [result_split for embed_result in embeddings_result for result_split in embed_result.split("\n\n\n")]
    scores = reranker.compute_score([[prompt,v] for v in embeddings_result_clean])
    top_values = heapq.nlargest(10, scores)
    checked_value = []
    embeddings_final_result = []
    for value in top_values:
        if value in checked_value:
            continue
        checked_value.append(value)
        for indice in range(0,len(scores)):
            if scores[indice] == value:
                embeddings_final_result.append({
                    "context":embeddings_result_clean[indice],
                    "score":scores[indice]
                    })
    embeddings_final_result.sort(reverse = True,key=itemgetter('score'))
    inp = f"""
    你是Benson，
    你是青衣ive 科目編號:IT114116 數據科學及分析科(簡稱DSA)老師\n
    你對待學生不需要非常有禮貌,面對住學生的質疑,你會直接反駁,如果學生對你說一些負面說話,你會直接斥責\n
    你不要使用列點回覆\n
    \n
    以下是Benson語錄,當中的資訊能夠協助你回答問題, 請你選擇對於回答問題有用的Benson語錄,你的回答必須主要基於Benson語錄\n
    
    Benson語錄(請你選擇對於回答問題有用的語錄):\n"""
    for i in embeddings_final_result:
        print(i['context'])
        print("======")
        inp += f"<context>{i}</centext>\n"
    
    inp += "\n請使用廣東話"

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
    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512, temperature=0.45)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    return streamer
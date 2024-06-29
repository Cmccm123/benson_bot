import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import chromadb
from vllm import LLM, SamplingParams
from threading import Thread

device = "cuda"  # the device to load the model onto
chroma_client = chromadb.PersistentClient(path="benson")

# Load models using vLLM
llm = LLM(model="Qwen/Qwen2-7B-Instruct", dtype=torch.float16, device=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
emb_model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh', trust_remote_code=True)

def generate_response(prompt):
    key_word_messages = [
        {"role": "system", "content": "你是關鍵字搜尋器,你需要在一句句子中選出關鍵字,這個關鍵字會用作做知識庫的搜尋,所以你應該要盡可能輸出所有可能的關鍵字,讓知識庫容易搜尋結果, 你只能輸出搜尋到關鍵字，也就是最終結果，不能輸出其他東西"},
        {"role": "user", "content": f"'你好，Google'的關鍵字是什麼"},
        {"role": "assistant", "content": "Google"},
        {"role": "user", "content": f"'你好，Google, 我是Tom'的關鍵字是什麼"},
        {"role": "assistant", "content": "Google,Tom"},
        {"role": "user", "content": f"'太郎你鍾意食咩"},
        {"role": "assistant", "content": "太郎,食"},
        {"role": "user", "content": f"'{prompt}'的關鍵字是什麼"}
    ]
    
    key_word_text = tokenizer.apply_chat_template(
        key_word_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate response using vLLM
    sampling_params = SamplingParams(max_tokens=512)
    outputs = llm.generate([key_word_text], sampling_params=sampling_params)
    response=""
    for output in outputs:
        response = output.outputs[0].text
    print(response)

    embeddings_result = []
    embeddings = emb_model.encode([response])
    collection = chroma_client.get_collection(name="benson")
    embeddings_result += collection.query(query_embeddings=embeddings.tolist(), n_results=6)['documents'][0]
    print(embeddings_result)

    inp = f"""
    你是Benson，
    你是青衣ive 科目編號:IT114116 數據科學及分析科(簡稱DSA)老師\n
    你對待學生不需要非常有禮貌\n
    你不要使用列點回覆\n
    \n
    我給你有關的訊息記錄,當中的資訊能夠協助你回答問題, 請你選擇對於回答問題有用的訊息記錄,基於訊息記錄模仿當中的風格,使用相同的立場進行作答\n
    訊息記錄:\n"""
    for i in embeddings_result:
        inp += f"<context>{i}</context>\n"
    inp += "\n請使用廣東話reply"

    messages = [
        {"role": "system", "content": inp},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    result = llm.generate([text], sampling_params=sampling_params)
    
    return result
#----
import discord
import os
from discord.ext import commands
import time
client = commands.Bot(command_prefix='/',intents=discord.Intents.all())


@client.command(description='input msg',name="benson")
async def msg(ctx, * ,input_msg: str):
    msg = "wait..."
    message = await ctx.send(msg)
    response = generate_response(input_msg)
    generated_text = ""
    for new_text in response:
        generated_text += new_text.outputs[0].text
        await message.edit(content=generated_text)
token = open("discord_token.txt", "r").read()
client.run(token)
import discord
import os
from model import generate_response
from discord.ext import commands
import time
client = commands.Bot(command_prefix='/',intents=discord.Intents.all())

user_chat = {}
tpt = {}

@client.command(description='input msg',name="benson")
async def msg(ctx, * ,input_msg: str):
    msg = "wait..."
    message = await ctx.send(msg)
    if ctx.message.author.id not in user_chat.keys():
        user_chat[ctx.message.author.id] = []
    response = generate_response(input_msg,user_chat[ctx.message.author.id],tpt.get(ctx.message.author.id,0.45))
    generated_text = ""
    time.sleep(1)
    time_start=time.time()
    for new_text in response:
        if new_text  != "":
            generated_text += new_text
            if time.time()-time_start>=1 and generated_text != "":
                await message.edit(content=generated_text)
                time_start=time.time()
    if time.time()-time_start < 1:
        time.sleep(1)
    if generated_text != "":
        await message.edit(content=generated_text)
    user_chat[ctx.message.author.id]+=[{"role": "user", "content": input_msg},
                                       {"role": "Benson", "content": generated_text}]


@client.command(description='reset benson memory',name="benson_reset")
async def msg(ctx):
    if ctx.message.author.id in user_chat.keys():
        user_chat[ctx.message.author.id] = []
    message = await ctx.send(f"<@{ctx.message.author.id}> reset done")

@client.command(description='set benson temperature,input floater',name="benson_tpt")
async def msg(ctx, * ,input_msg: str):
    try:
        tpt[ctx.message.author.id] = float(input_msg)
    except ValueError:
        message = await ctx.send(f"<@{ctx.message.author.id}> 我教過幾多次，set temperature要入數字呀...唔識又唔問，搞到最後錯晒先黎後悔...")
        return
    message = await ctx.send(f"<@{ctx.message.author.id}> done!")
    
token = open("discord_token.txt", "r").read()
client.run(token)
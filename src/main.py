import discord
import os
from model import generate_response
from discord.ext import commands
import time
client = commands.Bot(command_prefix='/',intents=discord.Intents.all())

user_chat = {}

@client.command(description='input msg',name="benson")
async def msg(ctx, * ,input_msg: str):
    msg = "wait..."
    message = await ctx.send(msg)
    if ctx.message.author.id not in user_chat.keys():
        user_chat[ctx.message.author.id] = []
    response = generate_response(input_msg,user_chat[ctx.message.author.id])
    generated_text = ""
    time_start=time.time()
    for new_text in response:
        if new_text  != "":
            generated_text += new_text
            if time.time()-time_start>=1:
                await message.edit(content=generated_text)
                time_start=time.time()
    await message.edit(content=generated_text)
    user_chat[ctx.message.author.id]+=[{"role": "user", "content": input_msg},
                                       {"role": "Benson", "content": generated_text}]
    
token = open("discord_token.txt", "r").read()
client.run(token)
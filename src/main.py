import discord
import os
from model import generate_response
from discord.ext import commands
import time
client = commands.Bot(command_prefix='/',intents=discord.Intents.all())


@client.command(description='input msg',name="benson")
async def msg(ctx, * ,input_msg: str):
    msg = "wait..."
    message = await ctx.send(msg)
    response = generate_response(input_msg)
    generated_text = ""
    time_start=time.time()
    for new_text in response:
        if new_text  != "":
            generated_text += new_text
            if time.time()-time_start>=1:
                await message.edit(content=generated_text)
                time_start=time.time()
    await message.edit(content=generated_text)
    
token = open("discord_token.txt", "r").read()
client.run(token)
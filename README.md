## What is this

it is a fun project for create a discord chatbot, which can imitate my my favorite teacher-Benson. the chatbot can talk with you as Benson. it is funny !

## How to start

1.  Create your discord bot in https://discord.com/developers/applications

2. You will get the token after you create the bot. Then input it in ```discord_token.txt```

3. Put ```benson_chat.json``` into data floder. the     format should be:
```json
[
    "恭喜!",
    "係地獄過我平時講既野",
    "D留言地獄過地獄",
]
```
You may ask CM/Sunny to get this file. Because it is involving Benson's privacy, so I don't push it to git.
    

4. Run Command

```bash
docker compose up
```
4. enjoy :)

## How it work

I am using RAG technical for this fun project. RAG is a amazing solution for llm to learn a pepole about memory and style.

for Role play in LLM. Efficient & performance of RAG is also over fine tuning

First I have careated a Embedding database and use a Embedding model which can let Benson' history chat data chang to vector, then solve to database. Script [bensonMsgToDB.py](script/bensonMsgToDB.py) is work for these

Second we can find useful information in Embedding database to make model to answer user question. I have a keyword searcher in the system(use LLM). When user ask a question. it will output a keyword to search Embedding database. After that we can use this keyword to find similar datas on Embedding database and pass these datas to LLM model. it is helpful for LLM to answer the question.


To be continued...

## TODO

Write todo list - not done 

## What is this

it is a fun project for create a discord chatbot, which can imitate my favorite teacher-Benson. the chatbot can talk with you as Benson. it is funny !

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
5. enjoy :) in discord, you can ues this command to chat with benson

```
/benson hi
```

## How it work

I am using RAG (Retrieval-Augmented Generation) techniques for this fun project. RAG is an amazing solution for enabling LLMs (Language Learning Models) to understand memory and style.

For role-playing in LLMs, the efficiency and performance of RAG often surpass fine-tuning.

First, I created an embedding database and used an embedding model. This model converts Benson's chat history data into vectors and stores them in the database. [bensonMsgToDB.py](script/bensonMsgToDB.py) handles this task.

Second, we can find useful information in the embedding database to help the model answer user questions. I have a keyword searcher in the system (using LLM). When a user asks a question, it outputs a keyword. We then use this keyword to find similar data in the embedding database and pass this data to the LLM model. This approach helps the LLM provide more accurate answers.


To be continued...

## TODO

Write todo list - not done 

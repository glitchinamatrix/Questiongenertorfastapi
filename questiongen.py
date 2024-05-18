from fastapi import FastAPI
from transformers import pipeline
import asyncio
import os

app = FastAPI()

# Disable symlinks warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load the pre-trained question answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", revision="626af31")


async def get_answer(paragraph, question):
	answer = qa_pipeline(question=question, context=paragraph)
	return {question: answer['answer']}


@app.get("/get_answers")
async def get_answers(paragraph: str, questions: list):
	tasks = [get_answer(paragraph, question) for question in questions]
	answers = await asyncio.gather(*tasks)
	return {question: answer for item in answers for question, answer in item.items()}

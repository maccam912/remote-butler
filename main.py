import logging
import os
from dataclasses import dataclass
from datetime import datetime

import nest_asyncio
from langchain import HuggingFacePipeline
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models import ChatOpenAI
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools.playwright.utils import create_async_playwright_browser
from telegram import Update
from telegram.ext import (ApplicationBuilder, ContextTypes, MessageHandler,
                          filters)

nest_asyncio.apply()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# response = agent_chain.run(input="Hi I'm Erica.")
# print(response)


def get_llm():
    return CTransformers(model="tiiuae/falcon-7b-instruct", lib="avx")
    # return HuggingFacePipeline.from_model_id(
    #     model_id="tiiuae/falcon-7b-instruct",
    #     task="text-generation",
    #     model_kwargs={"temperature": 0},
    # )
    # return ChatOpenAI(temperature=0)


def get_agent_chain():
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = browser_toolkit.get_tools()

    llm = get_llm()  # Also works well with Anthropic models
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            "memory_prompts": [chat_history],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
    )
    return agent_chain


@dataclass
class Butler:
    agent_chain: AgentType
    last_used_ts: datetime


class Butlers(dict):
    """
    A class that manages butlers for each chat.
    It is just a dictionary of chat_id to agent_chain,
    and if no butler exists one is created.
    If a butler exists but has not been used in a while,
    it is deleted and a new one created.
    It will let a user use square bracket notation.
    """

    butlers: dict = {}

    def __init__(self):
        super().__init__()

    def __getitem__(self, chat_id):
        if chat_id not in self.butlers:
            self.butlers[chat_id] = Butler(
                agent_chain=get_agent_chain(), last_used_ts=datetime.now()
            )
        else:
            butler = self.butlers[chat_id]
            if (datetime.now() - butler.last_used_ts).total_seconds() > 60 * 15:
                del self.butlers[chat_id]
                self.butlers[chat_id] = Butler(
                    agent_chain=get_agent_chain(), last_used_ts=datetime.now()
                )
            else:
                butler.last_used_ts = datetime.now()
        return self.butlers[chat_id]


butlers = Butlers()


async def butler_helper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(
        f"Received message for chat id {update.effective_chat.id}:  \
                 {update.message.text}"
    )
    # get chat that this is a part of
    chat_id = update.effective_chat.id
    # get butler for this chat
    butler: Butler = butlers[chat_id]

    response = await butler.agent_chain.arun(input=update.message.text)
    logging.info(f"Responding with: {response}")

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


if __name__ == "__main__":
    logging.info("Starting bot")
    application = ApplicationBuilder().token(os.environ.get("TELEGRAM_TOKEN")).build()
    logging.info("Created application")

    butler_helper = MessageHandler(filters.TEXT & (~filters.COMMAND), butler_helper)
    logging.info("Defined butler helper")

    application.add_handler(butler_helper)
    logging.info("Added butler helper to application")

    application.run_polling()

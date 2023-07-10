import logging
import os
from dataclasses import dataclass
from datetime import datetime

import nest_asyncio
from langchain.agents import AgentType, initialize_agent

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import SearxSearchWrapper


# from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

from langchain.tools.playwright.utils import create_async_playwright_browser
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

nest_asyncio.apply()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_llm(local: bool = False):
    # return CTransformers(model="TheBloke/mpt-30B-instruct-GGML", model_file="mpt-30b-instruct.ggmlv0.q8_0.bin", lib="avx")
    if not local:
        return ChatOpenAI(
            temperature=0.2,
            # openai_api_base="https://local-ai.k3s.koski.co/v1",
            # streaming=True,
            # request_timeout=1800,
        )
    else:
        return ChatOpenAI(
            temperature=0.2,
            openai_api_base="https://local-ai.k3s.koski.co/v1",
            streaming=True,
            request_timeout=1800,
        )


def get_agent_chain(local: bool = False):
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = browser_toolkit.get_tools()
    search = SearxSearchWrapper(searx_host="https://searxng.k3s.koski.co")
    tools += [
        Tool.from_function(
            func=search.run,
            name="Search",
            description="useful for when you need to answer questions that can be found on the internet"
            # coroutine= ... <- you can specify an async method if desired as well
        ),
    ]

    llm = get_llm(local)  # Also works well with Anthropic models
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
    local: bool

    def __init__(self, local: bool = False):
        super().__init__()
        self.local = local

    def __getitem__(self, chat_id):
        if chat_id not in self.butlers:
            self.butlers[chat_id] = Butler(
                agent_chain=get_agent_chain(self.local), last_used_ts=datetime.now()
            )
        else:
            butler = self.butlers[chat_id]
            if (datetime.now() - butler.last_used_ts).total_seconds() > 60 * 15:
                del self.butlers[chat_id]
                self.butlers[chat_id] = Butler(
                    agent_chain=get_agent_chain(self.local), last_used_ts=datetime.now()
                )
            else:
                butler.last_used_ts = datetime.now()
        return self.butlers[chat_id]


butlers = Butlers(local=False)
local_butlers = Butlers(local=True)


async def butler_helper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(
        f"Received message for chat id {update.effective_chat.id}:  \
                 {update.message.text}"
    )
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Received. Responding..."
    )
    # get chat that this is a part of
    chat_id = update.effective_chat.id
    # get butler for this chat
    butler: Butler = butlers[chat_id]

    prompt = update.message.text
    if prompt[0:5] == "local":
        prompt = prompt[5:]
        butler: Butler = local_butlers[chat_id]

    response = butler.agent_chain.run(input=update.message.text)
    logging.info(f"Responding with: {response}")

    await context.bot.edit_message_text(
        chat_id=update.effective_chat.id, message_id=msg.message_id, text=response
    )


if __name__ == "__main__":
    logging.info("Starting bot")
    application = ApplicationBuilder().token(os.environ.get("TELEGRAM_TOKEN")).build()
    logging.info("Created application")

    butler_helper = MessageHandler(filters.TEXT & (~filters.COMMAND), butler_helper)
    logging.info("Defined butler helper")

    application.add_handler(butler_helper)
    logging.info("Added butler helper to application")

    application.run_polling()

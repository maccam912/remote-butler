import logging
import os
from dataclasses import dataclass
from datetime import datetime

import nest_asyncio
import requests
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.llms import LlamaCpp

# from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools import Tool
from langchain.tools.playwright.utils import create_async_playwright_browser
from langchain.utilities import SearxSearchWrapper
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from tqdm import tqdm


def download_file(url, save_path):
    """
    Downloads a file from the given URL and saves it to the specified path with a progress bar.

    Parameters:
    - url (str): The URL of the file to download.
    - save_path (str): The path where the file should be saved.

    Returns:
    - str: A message indicating whether the file was downloaded or already exists.
    """

    if os.path.exists(save_path):
        return f"File already exists at {save_path}."

    response = requests.get(url, stream=True)

    if response.status_code != 200:
        return f"Failed to download the file. HTTP Status Code: {response.status_code}."

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(save_path, "wb") as file:
        for chunk in response.iter_content(block_size):
            t.update(len(chunk))
            file.write(chunk)
    t.close()

    if total_size != 0 and t.n != total_size:
        return "Error: Something went wrong with the download."

    return f"File downloaded and saved to {save_path}."


nest_asyncio.apply()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

download_file(
    "https://huggingface.co/TheBloke/OpenOrcaxOpenChat-Preview2-13B-GGML/resolve/main/openorcaxopenchat-preview2-13b.ggmlv3.q8_0.bin",
    "openorcaxopenchat-preview2-13b.ggmlv3.q8_0.bin",
)


def get_llm():
    return LlamaCpp(
        model_path="openorcaxopenchat-preview2-13b.ggmlv3.q8_0.bin",
        input={"temperature": 0.3, "max_length": 500, "top_p": 0.98},
    )
    # if model == "local":
    #     return ChatOpenAI(
    #         temperature=0.2,
    #         openai_api_base="https://local-ai.k3s.koski.co/v1",
    #         request_timeout=600,
    #     )
    # elif model == "3.5":
    #     return ChatOpenAI(
    #         temperature=0.2,
    #     )


def get_agent_chain():
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
    model: str

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
    msg = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Received. Responding..."
    )
    # get chat that this is a part of
    chat_id = update.effective_chat.id
    # get butler for this chat
    butler: Butler = butlers[chat_id]

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

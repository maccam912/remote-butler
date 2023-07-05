import logging
import os
from dataclasses import dataclass
from datetime import datetime

from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.tools.playwright.utils import create_sync_playwright_browser
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

chat_history = MessagesPlaceholder(variable_name="chat_history")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

sync_browser = create_sync_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = browser_toolkit.get_tools()

llm = ChatOpenAI(temperature=0)  # Also works well with Anthropic models
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

# response = agent_chain.run(input="Hi I'm Erica.")
# print(response)


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        """
        If the key is not in the dictionary, create a new butler.
        If the key is in the dictionary, 
        check if it has been used in the last 15 minutes.
        If it has not, delete it and create a new one.
        """

        if key not in self:
            logging.info(f"Creating new butler for chat {key}")
            self[key] = Butler(
                agent_chain=initialize_agent(
                    tools,
                    llm,
                    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    memory=memory,
                    agent_kwargs={
                        "memory_prompts": [chat_history],
                        "input_variables": [
                            "input",
                            "agent_scratchpad",
                            "chat_history",
                        ],
                    },
                ),
                last_used_ts=datetime.now(),
            )
        else:
            logging.info(f"Checking if butler for chat {key} is still fresh")
            if (datetime.now() - self[key].last_used_ts).seconds > 15 * 60:
                logging.info(f"Butler for chat {key} is stale, deleting and recreating")
                del self[key]
                self[key] = Butler(
                    agent_chain=initialize_agent(
                        tools,
                        llm,
                        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                        verbose=True,
                        memory=memory,
                        agent_kwargs={
                            "memory_prompts": [chat_history],
                            "input_variables": [
                                "input",
                                "agent_scratchpad",
                                "chat_history",
                            ],
                        },
                    ),
                    last_used_ts=datetime.now(),
                )
        return self[key].agent_chain


butlers = Butlers()


async def butler_helper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.info(f"Received message for chat id {update.effective_chat.id}:  \
                 {update.message.text}")
    # get chat that this is a part of
    chat_id = update.effective_chat.id
    # get butler for this chat
    butler: Butler = butlers[chat_id]

    response = butler.agent_chain.run(input=update.message.text)
    logging.info(f"Responding with: {response}")

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


if __name__ == "__main__":
    logging.info("Starting bot")
    application = ApplicationBuilder().token(os.environ.get("TELEGRAM_TOKEN")).build()

    butler_helper = MessageHandler(filters.TEXT & (~filters.COMMAND), butler_helper)

    application.add_handler(butler_helper)

    application.run_polling()

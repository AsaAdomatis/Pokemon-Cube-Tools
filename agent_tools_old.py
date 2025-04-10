""" A package for using generative tools to extract information about a card type"""

# pokemon imports
from pokemontcgsdk import Card
from pokemontcgsdk import Set
from pokemontcgsdk import Type
from pokemontcgsdk import Supertype
from pokemontcgsdk import Subtype
from pokemontcgsdk import Rarity

# langchain imports
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser, HumanMessage

from langchain.chains import LLMChain

# google gemini import
from langchain_google_genai import ChatGoogleGenerativeAI

# .env imports
import os
from dotenv import load_dotenv

# cleaning imports
import re
import json

class CardClassificationOutputParser(BaseOutputParser):
    def parse(self, text: str):

        clean = re.sub(r"```json|```", "", text).strip()
        try:
            data = json.loads(clean)
            return data
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from model output: {text}")

def get_card_category(card:Card):
    # Set the model name for our LLMs.
    GEMINI_MODEL = "gemini-1.5-flash"
    # Store the API key in a variable.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


    # Create the prompt template with the required variables.
    
    # getting the rules text DEBUG: will only work for specific formattings
    if len(card.rules) == 2:
        card_text = card.rules = card.rules[0]
    else:
        card_text = " ".join(card.rules[:-1])

    prompt = PromptTemplate.from_template("""
        You are a Pokemon TCG expert and further an expert at designing Pokemon TCG cubes. Given the effect text of a Trainer card, classify it as either
                            
        1. **Utility** - cards that provide specific situational value and do not provided any boost to a decks consistency.
        2. **Concistency** - cards that improve the chances of executing a strategy. Most consistency cards will draw you more cards, search your deck for a card, or let you select from a few. Consistency is further classified into:
            - **Draw**: cards that draw more cards.
            - **Search**: cards that let you search your deck for other cards.
            - **Pick**: cards that offer choices (e.g., look at X cards, choose one).
        3. **Both** - cards that provide specific situational value and also provide a consistency boost for a deck. These cards can also be further broken down using the same subtypes as concistency.

        ### Examples:
        Card Text: "Search your deck for a Pokémon, reveal it, and put it into your hand. Then, shuffle your deck."
        Output: "category": "Consistency", "subtype": "Search"

        Card Text: "Draw 3 cards."
        Output: "category": "Consistency", "subtype": "Draw"
                                          
        Card Text: "Discard your hand and draw 7 cards."
        Output: "category": "Consistency", "subtype": "Draw"

        Card Text: "Look at the top 7 cards of your deck. You may reveal a Supporter card you find there and put it into your hand. Shuffle the other cards back."
        Output: "category": "Consistency", "subtype": "Pick"

        Card Text: "Switch 1 of your opponent's Benched Pokémon with their Active Pokémon."
        Output: "category": "Utility", "subtype": null

        Card Text: "Flip a coin. If heads, search your deck for any 1 card and put it into your hand. Then, shuffle your deck."
        Output: "category": "Consistency", "subtype": "Search"
                                          
        Card Text: "Until the end of your opponent's next turn, each Pokémon in play, in each player's hand, and in each player's discard pile has no Abilities. (This includes cards that come into play on that turn.)"
        Output: "category": "Utility", "subtype": "null"       

        Card Text: "Choose 1: • Discard up to 3 cards from your hand. (You must discard at least 1 card.) If you do, draw cards until you have 5 cards in your hand. • Switch 1 of your opponent's Benched Pokémon V with their Active Pokémon."   
        Output: "category: "Both", "subtype": "Draw"

        Card Text: "Draw 3 cards. Switch out your opponent's Active Pokémon to the Bench. (Your opponent chooses the new Active Pokémon.)"
        Output: "category: "Both", "subtype": "Draw"

        ### Now classify this card         
        Card Text: "{card_text}"
                            
        Return a json object like this:
        "category": "Utility" or "Consistency", "subtype": "Draw" or "Search" or "Pick" or null
        """
    )


    llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0.3)
    classification_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=CardClassificationOutputParser()
    )


    result = classification_chain.run(card_text=card_text)

    return result

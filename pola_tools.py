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

# other imports
import pandas as pd

class CardClassificationOutputParser(BaseOutputParser):
    def parse(self, text: str):

        clean = re.sub(r"```json|```", "", text).strip()
        try:
            data = json.loads(clean)
            return data
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from model output: {text}")

class PolaTool():
    """ A class for using LLM generative tools to extract information about pokemon cards. 
    Pola comes from [Po]emon [La]rge Language Model"""

    def __init__(self, gemini_api_key:str=None, gemini_model:str="gemini-1.5-flash", ptcg_api_key:str=None):
        if gemini_api_key is None:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        elif gemini_api_key:
            self.gemini_api_key = gemini_api_key

        if ptcg_api_key is None:
            self.ptcg_api_key = os.getenv("POKEMONTCG_IO_API_KEY")
        elif ptcg_api_key:
            self.ptcg_api_key = ptcg_api_key

        self.gemini_model = gemini_model

    def classify_trainers(self, cards:list[Card]):
        """ Given a list of trainer cards from the PokemonTCG SDK, returns a list of classifications of those trainers"""

        # getting all card text into a list
        results = []
        for card in cards:
            if card.supertype != "Trainer":
                raise ValueError(f"PolaTool::classify_trainers: {card.name} is not a Trainer!")

            # extracting card text
            if len(card.rules) == 2:
                card_text = card.rules = card.rules[0]
            else:
                card_text = " ".join(card.rules[:-1])

            prompt = PromptTemplate.from_template("""
                You are a Pokemon TCG expert and further an expert at designing Pokemon TCG cubes. Given the card effect text of a Pokemon Trainer card, classify it as either:
                                    
                1. **Utility** - cards that provide specific situational value and do not provided any boost to a decks consistency.
                2. **Concistency** - cards that improve the chances of executing a strategy. Most consistency cards will draw you more cards, search your deck for a card, or let you select from a few. Consistency is further classified into:
                    - **Draw**: cards that draw more cards.
                    - **Search**: cards that let you search your deck for other cards.
                    - **Pick**: cards that offer choices (e.g., look at X cards, choose one).
                3. **Both** - cards that provide specific situational value and also provide a consistency boost for a deck. These cards can also be further broken down using the same subtypes as concistency.

                ## Examples:
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

                ## Now classify the following card:       
                Card Text: "{card_text}"
                                    
                Return a json object like this:
                "category": "Utility" or "Consistency", "subtype": "Draw" or "Search" or "Pick" or null
                """
            )

            llm = ChatGoogleGenerativeAI(google_api_key=self.gemini_api_key, model=self.gemini_model, temperature=0.3)
            classification_chain = LLMChain(
                llm=llm,
                prompt=prompt,
                output_parser=CardClassificationOutputParser()
            )

            result = classification_chain.run(card_text=card_text)
            results.append(result)

        return results
    
def cards_to_df(cards:list[Card]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # add cards to dataframe
    card_dict = {
        "id": [],
        "name": [],
        "supertype": [],
        "subtypes": [], # must be converted to string for storage
        "hp": [],
        "type": [], # same here
        "evolvesFrom": [],
        "weakness_types": [],
        "weakness_values": [],
        "convertedRetreatCost": [],
        "set_id": [],
        "number": [],
        "artist": [],
        "rarity": [],
        "flavorText": [],
        "small_image_url": [], # dealt with differently
        "large_image_url": [] # dealt with differently
    }

    abilities_dict = {
        "card_id": [],
        "name": [],
        "text": [],
        "type": []
    }

    attack_dict ={
        "card_id": [],
        "name": [],
        "text": [],
        "cost": [], # same here
        "convertedEnergyCost": [],
        "damage": []
    }

    # converting cards into dictonaries
    for card in cards:
        if card is not None:
            card_dict["id"].append(card.id)
            card_dict["name"].append(card.name)
            card_dict["supertype"].append(card.supertype)
            card_dict["subtypes"].append(', '.join(card.subtypes) if card.subtypes else None)
            card_dict["hp"].append(card.hp)
            card_dict["type"].append(', '.join(card.types) if card.types else None)
            card_dict["evolvesFrom"].append(card.evolvesFrom)

            # dealing with special case weaknesses
            weakness_types = [weakness.type for weakness in card.weaknesses] if card.weaknesses else None
            weakness_values = [weakness.value for weakness in card.weaknesses] if card.weaknesses else None
            card_dict["weakness_types"].append(', '.join(weakness_types) if weakness_types else None)
            card_dict["weakness_values"].append(', '.join(weakness_values) if weakness_values else None)

            card_dict["convertedRetreatCost"].append(card.convertedRetreatCost)
            card_dict["set_id"].append(card.set.id)
            card_dict["number"].append(card.number)
            card_dict["artist"].append(card.artist)
            card_dict["rarity"].append(card.rarity)
            card_dict["flavorText"].append(card.flavorText)
            card_dict["small_image_url"].append(card.images.small)
            card_dict["large_image_url"].append(card.images.large)

            # add abilities to abilities_dict
            if card.abilities:
                for ability in card.abilities:
                    abilities_dict["card_id"].append(card.id)
                    abilities_dict["name"].append(ability.name)
                    abilities_dict["text"].append(ability.text)
                    abilities_dict["type"].append(ability.type)
            
            # add attacks to attack_dict
            if card.attacks:
                for attack in card.attacks:
                    attack_dict["card_id"].append(card.id)
                    attack_dict["name"].append(attack.name)
                    attack_dict["text"].append(attack.text)
                    attack_dict["cost"].append(', '.join(attack.cost) if attack.cost else None)
                    attack_dict["convertedEnergyCost"].append(attack.convertedEnergyCost)
                    attack_dict["damage"].append(attack.damage)

    # converting to dataframes and displaying them
    cards_df = pd.DataFrame(card_dict)
    abilities_df = pd.DataFrame(abilities_dict)
    attacks_df = pd.DataFrame(attack_dict)
    return cards_df, abilities_df, attacks_df
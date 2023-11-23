import os
from langchain.llms import HuggingFaceHub

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

from langchain.chains import SequentialChain

from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('huggingface_api_key')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

model_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":0.2, "max_length":64})

def generate_restaurant_name_and_items(cuisine):

    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":0.2, "max_length":64})

    prompt_template_name = PromptTemplate(input_variables=["cuisine"],
                                        template="I want to open a {cuisine} resturant, suggest a fancy name for this"
                                        )

    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")


    llm = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature":0.2, "max_length":64})

    prompt_template_items = PromptTemplate(input_variables=["restaurant_name"],
                                        template="Suggest 5 food items which can be added in the menu for {restaurant_name}."
                                        )

    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=["cuisine"],
        output_variables=["restaurant_name", "menu_items"]
    )

    response = chain({"cuisine":"Arabic"})

    return response

if __name__=="__main__":
    print(generate_restaurant_name_and_items("Indian"))

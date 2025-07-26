"""
Module to label newspaper articles using OpenAI's GPT via LangChain.

Use this function in the manual labeling phase of a supervised NLP pipeline,
e.g., to label AI-related content for narrative indices.
"""

import os
from pathlib import Path
from typing import List, Union,Literal
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Define structured response model
class Response(BaseModel):
    label: Literal["Yes","No"]  


def prompt_articles(
    ver: Union[int, str],
    message_list: List[str],
    template: template,
    output_type: str = "pydantic",
    model_name: str = "gpt-4",
    temperature: float = 0.2,
) -> pd.DataFrame:
    """
    Prompts GPT model to label a list of newspaper article prompts.

    Parameters
    ----------
    ver : int or str
        Version of prompttemplate. 
    message_list : list of str
        Prompts/messages to send to the model.
    template : ChatPromptTemplate
        LangChain prompt template for formatting messages.
    output_type : str
        Either "pydantic" or "csv". Determines output parser.
    model_name : str
        GPT model to use (e.g., "gpt-4").
    temperature : float
        Sampling temperature for the LLM.

    Returns
    -------
    pd.DataFrame
        DataFrame containing prompts, raw and structured responses.
    """

    # Set up environment
    root = Path(__file__).resolve().parents[2]
    dotenv_loaded = load_dotenv(dotenv_path=root / ".env")
    api_key = os.getenv("API_KEY")

    if not dotenv_loaded or not api_key:
        print(".env file not loaded or API_KEY missing.")
        return pd.DataFrame()

    # Initialize LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model=model_name,
        temperature=temperature
    )

    # Define output parser
    parser = PydanticOutputParser(pydantic_object=Response, include_json_markdown=False)

    # Define prompt template
    full_prompt = template.partial(
        format_instructions=parser.get_format_instructions()
    )

    # Prompt loop
    results = []

    for i, msg in enumerate(message_list):
        try:
            prompt_msg = full_prompt.format_messages(article=msg)
            response = llm.invoke(prompt_msg)

            structured = parser.parse(response.content)

            results.append({
                "prompt_id": i,
                "prompt": msg,
                "structured_response": structured.label
            })

        except Exception as e:
            print(f"Error at prompt {i}: {e}")

    # write results to data frame and return
    df_final = pd.DataFrame(results)
    output_path = root / "data" / "processed" / "variables" / f"gpt_labeled_articles_{ver}.csv"
    df_final.to_csv(output_path, index=False)

    print(f"Results saved to {output_path}")
    return df_final
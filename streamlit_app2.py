"""Python file to serve as the frontend"""
import os
import streamlit as st
from streamlit_chat import message

from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.prompts.prompt import PromptTemplate

# Set up credentials
os.environ['OPENAI_API_KEY'] = st.secrets['api_keys']['openai']

graph = Neo4jGraph(
    url=st.secrets['neo4j_info']['uri'],
    username=st.secrets['neo4j_info']['user'],
    password=st.secrets['neo4j_info']['password'],
    database=st.secrets['neo4j_info']['database'],
)

# Get graph schema and engineer prompt
schema = graph.get_schema

CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
You are a bot that help answer questions about a knowledge graph, PrimeKG (Giant), in order to answer medical questions.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Always search for disease names and phenotype names using a case-insensitive search.
Never search for disease names or phenotype names using the curly braces syntax.
Always search for disease and phenotype names using the toLower() and CONTAINS functions
If you are unsure about the direction of a relationship arrow in the query, use an undirected relationship without a < or > character.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# What diseases have "Hypoglycemia" as a symptom?
MATCH (d:Disease)-[:PHENOTYPE_PRESENT]->(e:EffectOrPhenotype) WHERE toLower(e.name) CONTAINS 'hypoglycemia' RETURN DISTINCT d.name
# What drugs can be used to treat malaria?
MATCH (d:Disease)-[:INDICATION]->(dr:Drug) WHERE toLower(d.name) CONTAINS 'malaria' RETURN DISTINCT dr.name
# How many types of cancer are in the database?
MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'cancer' RETURN COUNT(DISTINCT d.name)

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

# Instantiate chain on app load
llm_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(temperature=0,
                          #openai_api_key=api_key,
                          model='gpt-4o-mini'),
    qa_llm=ChatOpenAI(temperature=0,
                      #openai_api_key=api_key,
                      model='gpt-3.5-turbo'),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT
)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Ask PrimeKG (Giant)", page_icon=":robot:")
st.header("Ask PrimeKG (Giant)")
st.markdown("""
    This bot allows you to query the giant component of a precision medicine knowledge graph,
    [Prime KG](https://www.nature.com/articles/s41597-023-01960-3) (Giant), using natural language.
    It uses langchain and OpenAI GPTs to generate Cypher queries to the Neo4j Aura database.
    
    Ask any question about the data contained in the knowledge graph.  Data questions may be somewhat keyword-sensitive,
    as the current database does not contain embeddings. Examples queries might include:
    - What does exposure to silica result in?
    - What drugs can be used to treat malaria? 
    - How many types of cancer are in the database?
    - What are the symptoms of bronchiolitis?
""")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Enter a question: ", "What types of nodes are in the database and how many of each kind are there? Show me the result as a table.", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = llm_chain.run({'query': user_input})

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

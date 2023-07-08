import numpy as np
import openai
import os
import pandas as pd
import re
from atlassian import Confluence
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack import WebClient
from slack_bolt import App


# load environment variables
load_dotenv()
user = os.getenv("USERNAME")
password = os.getenv("CONFLUENCE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# model settings
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = OPENAI_API_KEY
COMPLETIONS_API_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 500,
    "model": "text-davinci-003",
}

# Event API & Web API
app = App(token=SLACK_BOT_TOKEN) 
client = WebClient(SLACK_BOT_TOKEN)

regenerate_data = True

def remove_emojis(data):
    """Remove emojis from text data."""

    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def clean_content(content):
    """Parse text from HTML and clean it up"""
    
    soup = BeautifulSoup(content, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = remove_emojis(text)
    text = text.replace('\xa0', ' ')
    # add a space after .
    text = re.sub(r'\.(?! )', '. ', re.sub(r' +', ' ', text))
    text = re.sub(r'\\uD.*',' ',text)

    return text


def get_data(space_of_interest='IT KB Page (For demo)'):
    """Download relevant data from confluence space."""

    confluence = Confluence(
        url='https://tianma.atlassian.net/wiki',
        username = user,
        password = password,
    )   

    # get current global spaces
    spaces1 = confluence.get_all_spaces(start=0, limit=500, expand=None)
    spaces2 = confluence.get_all_spaces(start=500, limit=1000, expand=None)
    global_spaces = {}
    for space in spaces1['results']:
        if space['type'] == 'global' and space['status'] == 'current':
            global_spaces[space['name']] = space['key']
    for space in spaces2['results']:
        if space['type'] == 'global' and space['status'] == 'current':
            global_spaces[space['name']] = space['key']

    # get all pages from designated space
    pages = confluence.get_all_pages_from_space(global_spaces[space_of_interest], start=0, limit=100, status=None, expand=None, content_type='page')
    relevant_pages = {}
    for page in pages:
        if page['status'] == 'current':
            relevant_pages[page['title']] = page['id']

    # get content from pages and clean up text
    space_content = {}
    for key in relevant_pages:
        content = confluence.get_page_by_id(relevant_pages[key], "space,body.view,version,container")
        space_content[content['title']] = clean_content(content['body']['view']['value'])

    # prepare training dataframe
    df = pd.DataFrame(data={'title': space_content.keys(), 'content': space_content.values()})
    df['tokens'] = df.content.str.len()
    df = df[df['tokens']>100]
    # save df locally for later use
    df.to_csv('hackathon-data.csv',index=False)

    return df

def get_embedding(text, model='text-embedding-ada-002'):

    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        r.title: get_embedding(r.content) for idx, r in df.iterrows()
    }


def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query, contexts) :
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question, context_embeddings, df):
    """
    Fetch relevant context and add it to the prompt
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003
    separator_len = 3
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df[df['title']==section_index]
        chosen_sections_len += document_section.tokens.values[0] + separator_len
        if chosen_sections_len > 5000:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.values[0].replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    header = """Answer the question as truthfully as possible trying to use the provided context if possible. If the answer is not related to the text below say "That is outside of my expertise.""\n\nContext:\n"""
    header = header + "".join(chosen_sections)

    return header + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query,
    df,
    document_embeddings,
    show_prompt = False
):
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")



# This gets activated when the bot is tagged in a channel    
@app.event("app_mention")
def handle_message_events(body, logger):

    if regenerate_data:
        df = get_data('IT KB Page (For demo)')
    else:
        # read in data
        df = pd.read_csv('hackathon-data.csv')

    document_embeddings = compute_doc_embeddings(df)

    # Log message
    print(str(body["event"]["text"]).split(">")[1])
    
    # Create question for ChatGPT
    question = str(body["event"]["text"]).split(">")[1]
    
    # Let the user know that we are busy with the request 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"Hello from your bot! :robot_face: \nThanks for your request, I'm on it!")
    
    # Create prompt with context and get answer
    answer = answer_query_with_context(question, df, document_embeddings, False)
    
    # Reply to thread 
    response = client.chat_postMessage(channel=body["event"]["channel"], 
                                       thread_ts=body["event"]["event_ts"],
                                       text=f"{answer} \n\n\nIf the above cannot resolve your issue \nPlease log a ticket <https://tianma.atlassian.net/jira/software/c/projects/SD/boards/1|here> ")


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

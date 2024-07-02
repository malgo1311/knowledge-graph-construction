#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install openai
# !rm /work/pi_dhruveshpate_umass_edu/project_19/aishwarya/696DS-named-entity-extraction-and-linking-for-KG-construction/code/openai/credentials.json


# In[2]:


import os
import csv
import time
import json
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv


# In[3]:


load_dotenv("credentials.env")
api_key = os.getenv("api_key")


# In[4]:


client = OpenAI(api_key=api_key)
model = "gpt-4" # gpt-3.5-turbo-instruct gpt-4


# In[ ]:

output_filename = "sdapr25_entities.csv"

# ## Get Data

# In[5]:


data_path = "/work/pi_dhruveshpate_umass_edu/project_19/aishwarya/696DS-named-entity-extraction-and-linking-for-KG-construction/datasets/synthetic_data_Apr25"
file = "SD_Apr25_chunked_data.json"

with open(os.path.join(data_path, file), 'r') as f:
    data = json.load(f)


# In[7]:


# type(data), data.keys()



data_path = "/work/pi_dhruveshpate_umass_edu/project_19/ReDocREDPreprocessing/Re-DocRED/processed/"
file_v2 = "Re-DocRED_Processed_Train.csv"
    
def get_entity_example(idx, datatmp):
    allEntities = set()
    for item in datatmp["Triplets"][idx].split("\n"):
        item = item.split(" | ")
        if len(item) == 3:
            allEntities.add(item[0])
            allEntities.add(item[2])

    return datatmp["Text"][idx], '; '.join(list(allEntities))
    
def get_entities_prompt(text):
    
    data_v2 = pd.read_csv(os.path.join(data_path, file_v2), skiprows = range(1, 2000), nrows = 500)
    
    ex1, exout1 = get_entity_example(1, data_v2)
    ex2, exout2 = get_entity_example(2, data_v2)

    prompt=f'''Task: Please detect all the entities from the given input Text.
Entities could be people, organization, places, concepts, dates or any other proper nouns present in the text. \
Use the following examples as reference to understand the task. \
Give the output in the same format as given in the Example Entities Output, i.e., separated by a semicolon, ';'.

Example Text 1: {ex1}
Example Entities Output 1: {exout1}

Example Text 2: {ex2}
Example Entities Output 2: {exout2}

Text: {text}
Entities Output:'''
    
    return prompt

def get_system_msg():
    
    data_v2 = pd.read_csv(os.path.join(data_path, file_v2), skiprows = range(1, 2000), nrows = 500)
    
    ex1, exout1 = get_entity_example(1, data_v2)
    ex2, exout2 = get_entity_example(2, data_v2)

    prompt=f'''You are a helpful assistant and an expert in named entity extraction.
Task: Please detect all the entities from the given input Text.
Entities could be people, organization, places, concepts, dates or any other proper nouns present in the text. \
Use the following examples as reference to understand the task. \
Give the output in the same format as given in the Example Entities Output, i.e., separated by a semicolon, ';'.

Example Text 1: {ex1}
Example Entities Output 1: {exout1}

Example Text 2: {ex2}
Example Entities Output 2: {exout2}'''
    
    return prompt

def get_content(text):
    
    prompt=f'''Text: {text}
Entities Output:'''
    
    return prompt


# In[9]:


print(get_system_msg())
print(get_content("hey"))


# ### Relations Extraction

# In[8]:

get_entities_prompt("text")


# ## Predictions

# In[10]:


def get_prediction(text):
    
    prompt = get_entities_prompt(text)
    
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature = 0,
        max_tokens = 256,
    )
    time_diff = time.time() - start
    
    return response, time_diff

def get_chat_prediction(messages):
    
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = 0,
        max_tokens = 256,
    )
    time_diff = time.time() - start
    
    return response, time_diff


def get_relation_prediction(text, entities):
    
    prompt = get_relations_prompt(text, entities)
    
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    time_diff = time.time() - start
    
    return response, time_diff


# In[11]:


output_data = []


# ### Entity Extraction

# In[14]:


debug = False

all_keys = list(data.keys())

for i in tqdm(range(len(all_keys))):
    
    doc_key = all_keys[i]
    
    if doc_key != '5' and debug:
        continue
    
    # messages = [{"role": "system", "content": get_system_msg()}]
    
    for item in data[doc_key]:
        
        # print(messages)
    
        #
        text = item['chunk_text']
        chunk_idx = item['chunk_index']
        
        # messages.append({"role": "user", "content": get_content(text)})
        
        output, time_diff = get_prediction(text)
        # output, time_diff = get_chat_prediction(messages)
        # output, time_diff = get_relation_prediction(i)

        outputText = output.choices[0].message.content
        input_tokens = output.usage.prompt_tokens
        output_tokens = output.usage.completion_tokens
        
        # messages.append({"role": "assistant", "content": outputText})

        if debug:
            print(f"\nInput - {text}")
            # print(f"\nGT - {data['Triplets'][i]}")
            print(f"\noutput - {outputText}")

        output_data.append([doc_key, chunk_idx, outputText, time_diff,
                           input_tokens, output_tokens])

        if i % 10 == 0 and i > 1:

            with open(output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(output_data)
                
        # if debug: break
        
    if debug: break
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_data)







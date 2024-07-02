#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install openai
# !rm /work/pi_dhruveshpate_umass_edu/project_19/aishwarya/696DS-named-entity-extraction-and-linking-for-KG-construction/code/openai/credentials.json


# In[1]:


import os
import csv
import time
import json
import pandas as pd
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv


# In[2]:


load_dotenv("credentials.env")
api_key = os.getenv("api_key")


# In[3]:


client = OpenAI(api_key=api_key)
model = "gpt-4" # gpt-3.5-turbo-instruct gpt-4


# In[ ]:


output_filename = "sdapr25_relations_v2_3.csv"


# ## Get Data

# In[4]:


data_path = "/work/pi_dhruveshpate_umass_edu/project_19/aishwarya/696DS-named-entity-extraction-and-linking-for-KG-construction/"
file = "datasets/synthetic_data_Apr25/SD_Apr25_chunked_data.json"

with open(os.path.join(data_path, file), 'r') as f:
    data = json.load(f)


# In[7]:


# type(data), data.keys()


# ### RElations

# In[9]:


entities_data = pd.read_csv(os.path.join(data_path, "datasets/synthetic_data_Apr25/news_training_processed.csv"))
entities_data = entities_data.rename(columns={"Unnamed: 0": 'doc_key'})

print(entities_data.shape)


# In[11]:


data_json = {}
for idx in entities_data.index:
    doc_key = entities_data["doc_key"][idx]
    data_json[doc_key] = entities_data["processed_entities"][idx]


# ## Get prompt

# ### Relations Extraction

# In[8]:


data_path = "/work/pi_dhruveshpate_umass_edu/project_19/ReDocREDPreprocessing/Re-DocRED/processed/"
file_v2 = "Re-DocRED_Processed_Train.csv"
    
def get_example(idx, datatmp):
    allEntities = set()
    for item in datatmp["Triplets"][idx].split("\n"):
        item = item.split(" | ")
        # print(item, len(item))
        if len(item) == 3:
            allEntities.add(item[0])
            allEntities.add(item[2])

    allEntities = list(allEntities)
    # print(allEntities)
    # return datatmp["Text"][idx], allEntities, datatmp["Triplets"][idx]
    return datatmp["Text"][idx], "; ".join(allEntities), datatmp["Triplets"][idx]
    
def get_relations_prompt(text, entities):
    
    data_v2 = pd.read_csv(os.path.join(data_path, file_v2), skiprows = range(1, 2000), nrows = 500)
    
    ex1, exent1, exout1 = get_example(1, data_v2)
    ex2, exent2, exout2 = get_example(2, data_v2)

    prompt=f'''Task Description:
The task is to extract Relations between the Entity List for given text, in the form of triplets. \
Extract triplets from the given Text based solely on the relationships present in the text. \
Ensure that entities are chosen directly from the provided Entity List to maintain accuracy. \
Avoid duplicating triplets in the output. Use the provided Example Text and Relations Output as references \
to understand how to identify meaningful relationships between entities from Entity List. \
Pay attention to all potential relations between all the entities and include them in the output.

Example Text 1: {ex1}
Entity List of Text 1: {exent1}
Relations Output of Text 1: {exout1}

Example Text 2: {ex2}
Entity List of Text 2: {exent2}
Relations Output of Text 2: {exout2}

Text: {text}
Entity List: {entities}
Relations Output:'''
    
    return prompt


# In[9]:


print(get_relations_prompt("text", "entities"))


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


# ### Relation Extraction

# In[14]:

processed = set()

# In[20]:


debug = False

all_keys = list(data.keys())

for i in tqdm(range(len(all_keys))):
    
    doc_key = all_keys[i]
    
    if doc_key != '3' and debug:
        continue
        
    # if int(doc_key) <= 376 or int(doc_key) > 500:
    #     continue

    if int(doc_key) <= 911:
        continue
    
    # messages = [{"role": "system", "content": get_system_msg()}]
    
    for item in data[doc_key]:
        
        # print(doc_key, item)
        
        # print(messages)
    
        #
        text = item['chunk_text']
        chunk_idx = item['chunk_index']
        
        # print(doc_key, chunk_idx)
        
        
        # if (doc_key, chunk_idx) != ('3',0):
        #     continue
            
        # if (doc_key, chunk_idx) in processed:
        #     continue
        # else:
        #     processed.add((doc_key, chunk_idx))
        
        # if int(chunk_index) in data_json[int(doc_key)]:
        processed_entities = data_json[int(doc_key)]

        # messages.append({"role": "user", "content": get_content(text)})

        # output, time_diff = get_prediction(text)
        # output, time_diff = get_chat_prediction(messages)
        output, time_diff = get_relation_prediction(i, processed_entities)

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
            
        # else:
        #     output_data.append([doc_key, chunk_idx, "No data", 0,
        #                        0, 0])

        if i % 10 == 0 and i > 1:

            with open(output_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(output_data)
                
        if debug: break
        
    if debug: break
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_data)



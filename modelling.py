import json
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import pickle
import os
from sentence_transformers import SentenceTransformer
import re
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from transformers import pipeline

def extract_negated_claims(output):
    # Split the output by newline
    output_lines = output.split('\n')

    negated_claims = []

    # Check if there are split outputs that start with a bullet points, numbered points, or dashes
    if any(line.strip().startswith(('*', '•', '-', '–', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')) for line in output_lines):
        for line in output_lines:
            if line.strip().startswith(('*', '•', '-', '–', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                negated_claims.append(line.strip('•–*-12345678910.').strip())
        return negated_claims
    claims = []
    confirm_neg = []
    negation_terms = ['negate', 'negation', 'negated']

    for idx in range(1,len(output_lines)):
        line = output_lines[idx]
        if ":" in line:
            if not any(term in line.split(":")[0].lower() for term in negation_terms):

                if len(line.split(":")[1].strip())==0:
                    claims.append(output_lines[idx+1].strip()) 
                    idx+=1       
                else:
                    claims.append(":".join(line.split(":")[1:]).strip())
            else:
                if len(line.split(":")[1].strip())==0:
                    confirm_neg.append(output_lines[idx+1].strip()) 
                    idx+=1       
                else:
                    confirm_neg.append(":".join(line.split(":")[1:]).strip())
    if len(confirm_neg)==0:
        # print(claims)
        return claims
    else:
        # print(confirm_neg)
        return confirm_neg
    # return negated_claims


def isConclusion(section):
    
    if "conclusion" in section.lower() or "discussion" in section.lower():
        return True
    return False

def get_encodings(doc_no,sentences):
    enc = []
    for sentence in sentences:
        try:
            enc.append(encodings_dict[doc_no][sentence])
        except KeyError:
            enc.append(encoder.encode(sentence))
    return enc

# %%


def get_evidence(claim_encoding,evidence_encodings,top_n,evidences):
    similarities = cosine_similarity(claim_encoding.reshape(1, -1), evidence_encodings)[0]
    
    # Get indices of top N similar embeddings
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Get top N similar embeddings and their cosine similarity scores
    top_embeddings = [evidence_encodings[i] for i in top_indices]
    top_scores = [similarities[i] for i in top_indices]
    top_evidences = [evidences[i] for i in top_indices]

    return " ".join(top_evidences)

def get_dataset():
	dataset = []
	for fileno in tqdm(list(pos_claims.keys())[:10]):
		js_file = json.loads(open("Parsed_pds/%s.json"%fileno,"r").read())
		body_text = js_file['pdf_parse']['body_text']
		text = [x['text'] for x in body_text if not isConclusion(x['section'])]
		if len(js_file['pdf_parse']['abstract'])>0:
			text.insert(0,js_file['pdf_parse']['abstract'][0]['text'])
		text = " ".join(text)
		doc = nlp(text)
		sentences = [str(x) for x in doc.sents if len(x)>5]
		
		enc = get_encodings(fileno,sentences)
		claim_enc = encoder.encode(pos_claims[fileno])

		for idx,claim in enumerate(pos_claims[fileno]):
			ev = get_evidence(claim_enc[idx],enc,10,sentences)
			dataset.append([fileno,"positive",claim,ev])
        
		for idx,claim in enumerate(pos_claims[fileno]):
			ev = get_evidence(claim_enc[idx],enc,10,sentences)
			dataset.append([fileno,"positive",claim,ev])
		for positive_claim in neg_claims[fileno]:
			for x in extract_negated_claims(neg_claims[fileno][positive_claim][0]):
				ev = get_evidence(encoder.encode(x),enc,10,sentences)
				dataset.append([fileno,"negative",x,ev])
		#print(dataset)
	dataset = pd.DataFrame(dataset,columns=["FileNo","Type","Claim","Evidence"])
	return dataset

def T5_eval():
    pipe = pipeline('text2text-generation', model='google-t5/t5-base')
    prompt = '''verify claim: %s <sep> evidence: %s <sep>'''
    outputs = []
    
    for row in tqdm(list(dataset.itertuples())):
        claim = row[3]
        evidence = row[4]
        response = pipe(prompt%(claim,evidence),max_length=64,num_return_sequences=1)
        output = response[0]['generated_text']

        outputs.append(output)
    dataset["Output"] = outputs
    
    dataset.to_csv(output_file,index=False)
def llama_eval():
	pass

if __name__ == "__main__":
    #os.environ["HF_HOME"] = "/data/btp_data/Siddharth/huggingface_cache/"
    encoder = SentenceTransformer('all-mpnet-base-v2')
    nlp = spacy.load("en_core_web_sm")

    pos_claims = json.loads(open("positive_claims_extracted.json","r").read())
    neg_claims = json.loads(open('negative_claims_generated.json',"r").read())    
    
    try:
        dataset = pd.read_csv("Final_dataset.csv")
    except:
        
        dataset = get_dataset()
        dataset.to_csv("Final_dataset.csv",index=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model', type=str, required=True)
    parser.add_argument('-o', dest='output', type=str, required=True)
    parser.add_argument('-n', dest='num_examples', type=int, required=False)
    args = parser.parse_args()
    model = args.model
    output_file = args.output
    num_examples = len(dataset)
    if args.num_examples is not None:
        num_examples = args.num_examples
    dataset = dataset.iloc[:num_examples]
    model_to_func_mapping = {
    'T5':T5_eval,
    'llama':llama_eval
    }
    model_to_func_mapping[model]()

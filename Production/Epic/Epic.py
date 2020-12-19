from IPython import display
display.clear_output()
import warnings
warnings.filterwarnings("ignore")

from flask_ngrok import run_with_ngrok
from flask import Flask, request, url_for, redirect, request
from flask import render_template, render_template_string
import json
import re
import os
from pprint import pprint
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AutoModelForSequenceClassification
from transformers import pipeline


display.clear_output()
def gpu_check(gpu_info):
    gpu_info = '\n'.join(gpu_info)
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("\nRunning on device: ", str(device).upper())
    
    if gpu_info.find('failed') >= 0 and ram_gb < 30:
      print('\nSelect the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
      print('and then re-execute this cell.')
      print('\nTo enable a high-RAM runtime, select the Runtime > "Change runtime type"')
      print('menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
      print('re-execute this cell.')
      return False
    else:
      try:
          if gpu_info.find('failed') < 0:
             print(gpu_info)
             return False
      except:
        display.clear_output()
      finally:
        display.clear_output()
        print('\nYour runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))
        print('You are using a high-RAM runtime!')
        return True



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pbar = tqdm(total=11)
display.clear_output()
GPT2_MODEL_PATH = 'gpt2/'
GPT2_TOKENIZER_PATH = 'gpt2_tokenizer/'
gpt2_tokenizer = AutoTokenizer.from_pretrained(GPT2_TOKENIZER_PATH,
                                               model_max_length=1024,
                                               padding_side='right')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

pbar.update(1)
gpt2_model = AutoModelWithLMHead.from_pretrained(GPT2_MODEL_PATH, 
                                                 output_loading_info=False,
                                                 local_files_only=True,
                                                 pad_token_id=gpt2_tokenizer.eos_token_id).to(device)
pbar.update(1)
zeroshot_tokenizer = AutoTokenizer.from_pretrained('joeddav/xlm-roberta-large-xnli', verbose=False)
pbar.update(1)
zeroshot_model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli", output_loading_info=False).to(device)
pbar.update(1)
zeroshot_generator = pipeline("zero-shot-classification", device=0, model=zeroshot_model, tokenizer=zeroshot_tokenizer)
pbar.update(1)

sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", verbose=False)
pbar.update(1)
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", output_loading_info=False).to(device)
pbar.update(1)
sentiment_generator = pipeline("sentiment-analysis",model=sentiment_model, tokenizer=sentiment_tokenizer, return_all_scores=True, device=0)
pbar.update(1)

contradiction_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli", verbose=False)
pbar.update(1)
contradiction_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", output_loading_info=False).to(device)
pbar.update(1)
contradiction_generator = pipeline("sentiment-analysis",model=contradiction_model, tokenizer=contradiction_tokenizer, device=0, return_all_scores=True)
pbar.update(1)
pbar.close()


app = Flask(__name__, template_folder="templates", static_folder="static")
run_with_ngrok(app)   #starts ngrok when the app is run


literary_tokens = ["characterization", "character", "setting", 
                   "exposition", "climax", "resolution", "plot", 
                   "context", "action", "weapon", 
                   "danger", "death", "suspense", 
                   "emotion", "surprise", "problem", "conflict", 
                   "perspective", "transition", "relief", "metaphor", 
                   "flashback"]

subgenre_tokens = ['Vampire', 'Ghost', 'Horror', 'Comedic Horror', 'Murder', 
                   'Werewolf', 'Apocalypse','Haunted House', 'Witch', 'Hell', 
                   'Alien', 'Gore', 'Monster']


author_tokens = {'Clive Barker' : '[CLIVE BARKER]', 'J. K. Rowling' : '[J.K. ROWLING]', 'Stephen King' : '[STEPHEN KING]', 'ThÃƒÆ’Ã‚Â©ophile Gautier' : '[THEOPHILE GAUTIER]', 
            'James H. Hyslop' : '[JAMES H HYSLOP]', 'Lord Edward Bulwer-Lytton' : '[LORD EDWARD BULWER-LYTTON]', 'A. T. Quiller-Couch' : '[A. T. QUILLER-COUCH]', 
            'Mrs. Margaret Oliphant' : '[MRS. MARGARET OLIPHANT]', 'Ernest Theodor Amadeus Hoffmann' : '[ERNEST THEODOR AMADEUS HOFFMAN]', 'Erckmann-Chatrian' : '[ERCKMANN-CHATRAIN]', 
            'Fiona Macleod' : '[FIONA MACLEOD]', 'Amelia B. Edwards' : '[AMELIA B. EDWARDS]', 'H. B. Marryatt' : '[H. B. MARRYATT]', 'Thomas Hardy' : '[THOMAS HARDY]', 
            'Montague Rhodes James' : '[MONTAGUE RHODES JAMES]', 'Fitz-James O\'Brien' : '[FITZ-JAMES O\'BRIEN', 'James Stephen' : '[JAMES STEPHEN]', 'Alfred Lord Tennyson' : '[ALFRED LORD TENNYSON]',
            'Amelia Edwards' : '[AMELIA EDWARDS]', 'Edward Bulwer-Lytton' : '[EDWARD BULWER-LYTTON]', 'Erckmann Chatrian' : '[ERCKMANN CHATRIAN]', 'Latifa al-Zayya' : '[LATIFA AL-ZAYYA]',
            'M. R. James' : '[M. R. JAMES]', 'Paul Brandis' : '[PAUL BRANDIS]', 'Brain Evenson' : '[BRAIN EVENSON]', 'Elliott O\'Donnell' : '[ELLIOTT O\'DONNELL]', 
            'Joseph, Sheridan Le Fanu' : '[JOSEPH, SHERIDAN LE FANU]', 'Edgar Allan Poe' : '[EDGAR ALLEN POE]', 'Bram Stoker' : '[BRAM STOKER]', 'Algernon Blackwood' :'[ALGERNON BLACKWOOD]',
            'Miles Klee' : '[MILES KLEE]', 'Nnedi Okorador' : '[NNEDI OKORADOR]', 'Sofia Samatar' : '[SOFIA SAMATAR]', 'Franz Kafka' : '[FRANZ KAFKA]', 'Laird Barron' : '[LAIRD BARRON]',
            'Nathan Ballingrud' : '[NATHAN BALLINGRUD]', 'Nellie Bly' : '[NELLIE BLY]', 'William Hop Hodgson' : '[WILLIAM HOP HODGSON]', 'Ambrose Bierce' : '[AMBROSE BIERCE]',
            'Kelly Link' : '[KELLY LINK]', 'Arthur Machen' : '[ARTHUR MACHEN]', 'George Sylvester Viereck' : '[GEORGE SYLVESTER VIERECK]', 'Robert Chambers' : '[ROBERT CHAMBERS]',
            'John Meade Falkner' : '[JOHN MEADE FALKNER]', 'Ann Radcliffe' : '[ANN RADCLIFFE]', 'Howard Lovecraft' : '[HOWARD LOVECRAFT]', 'Louis Stevenson' : '[LOUIS STEVENSON]',
            'Edith Birkhead' : '[EDITH BIRKHEAD]', 'Jeff Vandermeer' : '[JEFF VANDERMEER]', 'Henry James' : '[HENRY JAMES]', 'John William Polidori' : '[JOHN WILLIAM POLIDORI]',
            'Bob Holland' : '[BOB HOLLAND]', 'Oliver Onions' : '[OLIVER ONIONS]'}

def generate_checkbox(val, classes):
    return f'''
    <div class="checkbox-container {classes}">
        <span class="input-title">{val}</span>
        <label class="checkbox-label">
            <input type="checkbox" value="{val}">
            <span class="checkbox-custom rectangular"></span>
        </label>
    </div>
    '''

def generate_inputs():
    author_inputs = f''''''
    for author in author_tokens:
        author_inputs += generate_checkbox(author, "author")
    

    genre_inputs = f''''''
    for genre in subgenre_tokens:
        genre_inputs += generate_checkbox(genre, "genre")
        
    return author_inputs, genre_inputs

def get_blacklist_inputs(blacklist_path="Black_List.txt"):
    blacklist_words = f''''''
    with open(blacklist_path, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            blacklist_words += f'''<li class="blacklist" data-word="{line.lstrip().rstrip()}"><i class="fas fa-minus-circle delete"></i></li>\n'''
    return blacklist_words

def classify_input_tokens(input_text, threshold=0.9):
    tokens=[]
    #classify the lines according to literary tokens
    literary_generator = zeroshot_generator(input_text, literary_tokens, multi_class=True)
    for i, score in enumerate(literary_generator['scores']):
        if score > threshold:
            tokens.append("[" + literary_generator['labels'][i].upper() + "]")
        else: break
    return tokens


def get_bad_word_ids(bad_words=None, blacklist_path="Black_List.txt"):
    if not bad_words:
        bad_words = []
        with open(blacklist_path, "r") as reader:
            lines = reader.readlines()
            for word in lines:
                if (len(word) > 0):
                    bad_words.append(word)
    bad_word_ids = [gpt2_tokenizer.encode(bad_word) for bad_word in bad_words]
    return bad_word_ids

def remove_special_tokens(input):
    token_pattern = r"[^[]*\[([^]]*)\]"
    input = re.sub(token_pattern, "", input)
    return input

def find_dialogue_locs(text):
    dialogue_locations = []
    dialogue_pattern = r'"(?:(?:(?!(?<!\\)").)*)[.?!,]"'
    for match in re.finditer(dialogue_pattern, text):
        s = match.start()
        e = match.end()
        dialogue_locations.append((s, e))
    return dialogue_locations

def honorific_found(text, index):
    pat_obj = re.compile('(Mr)|(Mrs)|(Dr)|(Ms)|(Sr)|(Jr)|(Mt)', re.IGNORECASE)
    if pat_obj.search(text[index-4: index]):
        return True
    return False

def dialogue_found(locs, index):
    for s, e in locs:
        if (s <= index) and (index <= e):
            return True
    return False

def find_sentence_locs(text):
    #find and store locations of quotations within text 
    dialogue_locations = find_dialogue_locs(text)
    punc_locations = []
    for match in re.finditer("[!.?]", text):
        punc_i = match.end()
        if honorific_found(text, punc_i):
            continue      
        if dialogue_found(dialogue_locations, punc_i):
            continue
        punc_locations.append(punc_i)
    return punc_locations

def generate_outputs(sequence, context_input, bad_words_ids, num_sequences=3, top_k=50, top_p=0.97, max_length=1024, temperature=0.8):
    input_ids = gpt2_tokenizer.encode(sequence, return_tensors='pt').to(device)  # encode input context
    sample_outputs = gpt2_model.generate(
        input_ids,
        do_sample=True, 
        max_length=max_length, 
        top_k=top_k, 
        top_p=top_p,
        temperature=temperature,
        no_repeat_ngram_size=4,
        num_return_sequences=num_sequences, 
        bad_words_ids=bad_words_ids,
        early_stopping=True
    )
    decoded_outputs = []
    for i, sample_output in enumerate(sample_outputs):
        output = remove_special_tokens(gpt2_tokenizer.decode(sample_output, skip_special_tokens=True))
        output = output.replace(context_input, "").strip()
        decoded_outputs.append(output)

    return decoded_outputs

def print_remove(outputs, width=80):
    for output in outputs:
        pprint(remove_special_tokens(output), width=width)

def get_max_tokenizer(input, context):
    gpt2_tokens = gpt2_tokenizer.tokenize(context+input)
    contradiction_tokens = contradiction_tokenizer.tokenize(context+input)
    sentiment_tokens = sentiment_tokenizer.tokenize(context+input)
    max_tokenizer = gpt2_tokenizer
    max = len(gpt2_tokens)
    if len(contradiction_tokens) > max:
        max_tokenizer = contradiction_tokenizer
        max = len(contradiction_tokens)
    if len(sentiment_tokens) > max:
        max_tokenizer = sentiment_tokenizer
        max = len(sentiment_tokens)
    return max_tokenizer

def get_new_context(context, num_sentences=5):
    context_puncs = find_sentence_locs(context)
    num_puncs = num_sentences + 1
    if len(context_puncs) >= num_puncs:
      context = context[context_puncs[-num_puncs]:]
    return context

def get_sequence(input, context, tokens):
    max_tokens = 510
    #build a string of user defined tokens
    tokens = " ".join(tokens)
    max_tokenizer = get_max_tokenizer(input, context)
    context_input_tokenized = max_tokenizer.tokenize(context+" "+input)
    user_tokens_tokenized = gpt2_tokenizer.tokenize(tokens)
    if len(context_input_tokenized + user_tokens_tokenized) > max_tokens:
        index_for_slice = max_tokens - len(user_tokens_tokenized)
        context_input_tokenized = context_input_tokenized[-index_for_slice:]
        tokens = gpt2_tokenizer.convert_tokens_to_string(user_tokens_tokenized)
        context_input = max_tokenizer.convert_tokens_to_string(context_input_tokenized)
        return  tokens + context_input, context_input
    else:
        context_input = context + " " + input
        return tokens + context_input, context_input

def get_sentiment(sequence):
    sequence_max_score = 0
    max_sentiment = ""
    for d in sentiment_generator(sequence)[0]:
      if d['score'] > sequence_max_score:
          sequence_max_score = d["score"]
          max_sentiment = d["label"]
    return int(max_sentiment.split(" ")[0]), sequence_max_score

def get_contradiction(sequence, output):
    sequence_tokens = contradiction_tokenizer.tokenize(sequence + " " + output)
    sequence = contradiction_tokenizer.convert_tokens_to_string(sequence_tokens[-510:])
    d = contradiction_generator(sequence)[0][0] 
    return d["score"] 

def clean_output(output):  
    output = output.replace("\n", "")
    output = output.replace("newline>", "")
    #round off to last sentence, if possible
    punc_locations = find_sentence_locs(output)
    output = output[:punc_locations[-1]] if len(punc_locations) > 0 else output
    return output

def get_accepted_outputs(sequence, outputs):
    sequence_sentiment, sequence_score = get_sentiment(sequence)
    accepted = []
    for output in outputs:
        if len(output) == 0:
            continue 
        output = clean_output(output)
        output_sentiment, output_score = get_sentiment(output)
        if abs(output_sentiment - sequence_sentiment) <= 1:
            output_contradiction = get_contradiction(sequence, output)
            if output_contradiction < 0.8:
                accepted.append((output, output_score))
    return accepted

def get_best_output(outputs):
    max = -1
    output = ""
    for sample, score in outputs:
        if score > max:
            max = score
            output = sample
    return output

@app.route("/")
def home():
    author_inputs, genre_inputs = generate_inputs()
    blacklist_inputs = get_blacklist_inputs()
    return render_template("index.html", author_inputs=author_inputs, genre_inputs=genre_inputs, blacklist_inputs=blacklist_inputs)
        

@app.route("/settings", methods=['POST'])
def settings():
    global CONTEXT, TEMPERATURE, GENERATION_LENGTH, BLACKLIST, USER_TOKENS
    data = request.get_json()
    USER_TOKENS = data['tokens']
    CONTEXT = data['context']
    BLACKLIST = data['blacklist']
    TEMPERATURE = data['temperature']
    GENERATION_LENGTH = data['generation_length']
    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}

@app.route("/generate", methods=['POST'])
def generate():
    global CONTEXT, TEMPERATURE, GENERATION_LENGTH, BLACKLIST, USER_TOKENS
    data = request.get_json()

    CONTEXT = data['context']
    if CONTEXT[:-1] != " ": CONTEXT += " "

    new_input = data['input']
    new_context = get_new_context(CONTEXT, num_sentences=5)

    sequence, context_input = get_sequence(new_input, 
                                           new_context, 
                                           USER_TOKENS+classify_input_tokens(new_context))

    bad_words_ids = get_bad_word_ids(bad_words=BLACKLIST)
    
    accepted_outputs = []
    sequence_length = len(gpt2_tokenizer.tokenize(sequence))
    while (len(accepted_outputs) < 1):
        predicted_outputs = generate_outputs(sequence, 
                                             context_input, 
                                             bad_words_ids[:-1], 
                                             num_sequences=3, 
                                             top_k=150, 
                                             top_p=0.99, 
                                             temperature=float(TEMPERATURE), 
                                             max_length=sequence_length+int(GENERATION_LENGTH))
      
        accepted_outputs = get_accepted_outputs(context_input, predicted_outputs)

    output = get_best_output(accepted_outputs)
    output = new_input + " " + output
    output = " ".join(output.split())

    if output.find(CONTEXT) != -1: CONTEXT = output
    else: CONTEXT += output
    response = app.response_class(
        response=json.dumps({'context': CONTEXT}),
        status=200,
        mimetype='application/json'
    )

    return response


def run():
    display.clear_output()
    CONTEXT = ''
    TEMPERATURE = 0.75
    GENERATION_LENGTH = 20
    BLACKLIST = []
    USER_TOKENS = []
    bad_words_ids = get_bad_word_ids()

    # Application is running on http://_________.ngrok.io  
    app.run()

run()

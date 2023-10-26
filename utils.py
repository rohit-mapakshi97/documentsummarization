#!/usr/bin/python3
import csv
import pandas as pd 


def separateSummary():
    with open('./data/prepocessed.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        for row in reader:
            article = row[2]
            summary = row[3]
            # Write the article and summary to separate files
            with open('./data/articles.txt', 'a', encoding='utf-8') as fa, open('./data/summaries.txt', 'a', encoding='utf-8') as fs:
                fa.write(article + '\n')
                fs.write(summary + '\n')

# separateSummary()


def copy3000():
    df = pd.read_csv('./data/prepocessed.csv', delimiter=',',
                     engine='python', error_bad_lines=False, nrows=3000)
    df.to_csv('./data/prepocessed_3000.csv', index=False)
    
def saveGPT2(): 
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    # Load the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Save the model weights
    model.save_pretrained('./models/gpt2model')
    tokenizer.save_pretrained('./models/gpt2tokenizer')

def testGPT2():
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('./models/gpt2tokenizer')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('./models/gpt2model')

    # Generate a summary
    original_text = ""
    input_ids = tokenizer.encode(original_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=30, do_sample=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    print(summary)

    # input_ids = tokenizer.encode(input_text, return_tensors='pt')
    # output_ids = model.generate(input_ids, max_length=104, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
    # output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # print(output_text)

# testGPT2()

def testBART():
    import pickle
    # Load the saved model
    with open("./models/fineTunedBert.pkl", "rb") as f:
        saved_model = pickle.load(f)

    # Use the loaded model to make predictions or perform other tasks
    text = saved_model.predict("anything else stop sum artist think translate online profile words Twitter allows entire page indulgence website would allow Bring salient features creativity experience passion reasons painting Make clear readers artist loves art produces high quality art true champion art great words find friend help really important aspect selling online establishment credibility reliability")
    print(text)

testBART()
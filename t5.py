#!/usr/bin/python3
import pandas as pd 
from simplet5 import SimpleT5
from rouge import Rouge

def fine_tune_and_evaluate(): 
    # Data Prep 
    df = pd.read_csv('./data/prepocessedALL.csv', delimiter=',',
                    engine='python', error_bad_lines=False, nrows=3000)

    print(df.info())

    drop_cols = ['overview', 'sectionLabel', 'title']
    df = df.drop(drop_cols, axis=1)
    df = df.dropna()

    df.rename(columns={"headline":"target_text", "text": "source_text"}, inplace= True)
    print(df.info())

    # T5 model expects a task related prefix: since it is a summarization task, we have to add prefix "summarize: "
    df['source_text'] = "summarize: " + df['source_text']
    print(df.head(1)['source_text'])

    # Train, Test, Val split (60, 20, 20)
    train_data = df.sample(frac=0.60) #60%
    rest_part_40 = df.drop(train_data.index)
    test_data = rest_part_40.sample(frac=0.50) #20%
    validation_data = rest_part_40.drop(test_data.index) #20%
    print("Shapes: ", train_data.shape, validation_data.shape, test_data.shape)

    # Finetuning T5 model
    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")

    model.train(train_df=train_data,
                eval_df=validation_data,
                source_max_token_len=150,
                target_max_token_len=64,
                outputdir = "./models/",
                max_epochs=3, use_gpu=False)

    # Evaluation 
    model_t5 = SimpleT5()
    model_t5.load_model("t5","./models/simplet5-epoch-0-train-loss-2.7284-val-loss-2.2741", use_gpu=False)

    def predict_summary(row):
        input_text = row["source_text"] # assuming your DataFrame column is named "input_text"
        summary = model_t5.predict(input_text, max_length=512)
        return summary
    test_data["predicted_summary"] = df.apply(predict_summary, axis=1)
    df = df.dropna()
    # print(type(test_data["predicted_summary"]))
    rouge = Rouge()
    scores = rouge.get_scores(test_data["target_text"], test_data["predicted_summary"])
    print(scores)

def summarizeT5(text, model_t5):     
    input_text = "title: " + text
    output_title = model_t5.predict(input_text, max_length=16)

    input_text = "outline: " + text
    output_outline = model_t5.predict(input_text, max_length=512)

    input_text = "summarize: " + text
    output_summary = model_t5.predict(input_text, max_length=1024)
      
    return output_title, output_outline, output_summary 


if __name__ == "__main__": 
    fine_tune_and_evaluate()
    
    # text = "anything else stop sum artist think translate online profile words Twitter allows entire page indulgence website would allow Bring salient features creativity experience passion reasons painting Make clear readers artist loves art produces high quality art true champion art great words find friend help really important aspect selling online establishment credibility reliability"
    # model_t5 = SimpleT5()
    # model_t5.load_model("t5","./models/simplet5-epoch-0-train-loss-2.7284-val-loss-2.2741", use_gpu=False)
    # print(summarizeT5(text, model_t5))
    # pass

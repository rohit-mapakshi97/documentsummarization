import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel,Seq2SeqArgs
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import pipeline
import pickle


def finetune(): 
    # Dataprep 
    df = pd.read_csv('./data/prepocessed_3000.csv', delimiter=',',
                    engine='python', error_bad_lines=False, nrows=3000)

    print(df.info())

    drop_cols = ['overview', 'sectionLabel', 'title']
    df = df.drop(drop_cols, axis=1)
    df = df.dropna()

    df.rename(columns={"headline":"target_text", "text": "input_text"}, inplace= True)
    print(df.info())

    print(df.head(1)['input_text'])


    model_args = Seq2SeqArgs()
    model_args.num_train_epochs = 3
    model_args.no_save = True
    # model_args.per_device_train_batch_size=16,  # batch size per device during training
    # model_args.per_device_eval_batch_size=64,   # batch size for evaluation
    model_args.logging_dir='./logs',            # directory for storing logs
    model_args.evaluate_generated_text = True
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.max_length = 600
    model_args.overwrite_output_dir = True

    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-large",
        args=model_args,
        use_cuda=False,
    )

    train, test = train_test_split(df, test_size=0.2, random_state=23, shuffle=True)

    # finetune 
    model.train_model(train, eval_data=test)

    results = model.eval_model(test)
    print(results)

    with open("./models/fineTunedBert_local.pkl", "wb") as f:
        pickle.dump(model, f)

model = pipeline("summarization", model="facebook/bart-large-cnn")

def summarizeBART(text): 
    output_title = model(text, max_length=16, min_length=5, do_sample=False)

    output_outline = model(text, max_length=512, min_length=50, do_sample=False)

    output_summary = model(text, max_length=1024, min_length=100, do_sample=False)
      
    return output_title, output_outline, output_summary 

# text = '''photographer keep necessary lens cords batteries quadrant home studio Paints kept brushes cleaner canvas print supplies ink etc Make broader groups areas supplies make finding easier limiting search much smaller area ideas include Essential supplies area things use every day Inspiration reference area Dedicated work area Infrequent secondary supplies area tucked way mean cleaning entire studio means keeping area immediately around desk easel pottery wheel etc clean night Discard trash unnecessary materials wipe dirty surfaces Endeavor leave workspace way sit next day start working immediately without work tidying Even rest studio bit disorganized organized workspace help get business every time want make art visual people lot artist clutter comes desire keep track supplies visually instead tucked sight using jars old glasses vases cheap clear plastic drawers keep things sight without leaving strewn haphazardly ideas beyond mentioned include Canvas shoe racks back door Wine racks cups slot hold pens pencils Plastic restaurant squirt bottles paint pigment etc Simply string wires across wall along ceiling use hold essential papers want cut ruin tacks tape Cheap easy also good way handle papers ideas touch regularly need pin inspiration Shelving artist s best friend cheap easy way get room studio art space afraid get high either especially infrequently used supplies upper reaches room often under utilized provide vital space tools materials Turning one wall chalkboard gives perfect space ideas sketches planning without requiring extra equipment space even use smaller areas Paint jars storage equipment allowing relabel chalk needs change lot disorganization comes keep moving location things trying optimize space reorganizing frequently usually opposite effect leading lost items uncertainty cleaning afternoon label maker solve everything Instead spending mental energy looking storing things follow labels freeing mind think art month purge studio essential part project either throw file away later Artists constantly making new things experimenting making mess good thing set aside time declutter may fun moment lot fun spending 30 minutes digging junk find right paint old sketch sentimental used last six months little chance use next six months Toss'''

# print(summarizeBART(text))



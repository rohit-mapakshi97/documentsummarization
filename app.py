from flask import Flask, render_template, request
from simplet5 import SimpleT5
from t5 import summarizeT5
from textrank import textrank_summarization
from bart import summarizeBART
app = Flask(__name__)

# Loading T5 Model
model_t5 = SimpleT5()
model_t5.load_model("t5", "models/simplet5-epoch-2-train-loss-4.0892-val-loss-4.3157", use_gpu=False)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/summarize', methods=['POST', 'GET'])
def summarize():
    text = request.form['text']
    model_name = request.form['model']
    output_summary, output_outline, output_title = None, None, None
    if model_name == 't5':
        output_title, output_outline, output_summary = summarizeT5(
            text, model_t5)
        output_summary = output_summary[0]
        output_outline = output_outline[0]
        output_title = output_title[0]
        pass
    if model_name == 'textrank':
        output_summary = textrank_summarization(text)
        pass
    if model_name == 'bart':
        output_title, output_outline, output_summary = summarizeBART(text)
        output_title = output_title[0]['summary_text']
        output_outline = output_outline[0]['summary_text']
        output_summary = output_summary[0]['summary_text']
        pass
    return render_template('index.html', title=output_title, outline=output_outline, summary=output_summary)


if __name__ == '__main__':
    app.run(debug=True)

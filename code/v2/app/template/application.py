from flask import Flask,request,render_template
import pandas
import numpy
import nltk
import random

nltk.download('stopwords')
nltk.download('inaugural')  
nltk.download('punkt')     
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.corpus import inaugural
from nltk.tokenize import sent_tokenize
import re
import os

pattern = r"""(?x)
    (?:[A-Z]\.)+
    |\d+(?:\.\d+)?%?
    |\w+(?:[-']\w+)*
    |\.\.\.
    |(?:[.,;"'?():-_`])
    """
stopworddic=set(stopwords.words('english'))
concdict=pandas.read_csv("concdict.csv")
freqdict=pandas.read_csv("freqdict.csv")
percentile=pandas.read_csv("percentile.csv")
percentile_front=pandas.read_csv("percentile_front.csv")


def is_passive_voice(sentence):
    # determine if a sentence is (probably) in "active" or "passive" voice
    # return 1 if active, 0 if passive, -1 if indeterminate (rare)

    if len(nltk.sent_tokenize(sentence)) > 1:
        return None

    tags0 = numpy.asarray(nltk.pos_tag(nltk.word_tokenize(sentence)))
    try:
        tags = tags0[numpy.where(~numpy.in1d(tags0[:, 1], ['RB', 'RBR', 'RBS', 'TO', ]))]  # remove adverbs, 'TO'
    except IndexError:
        return None

    if len(tags) < 2:  # too short to really know.
        return False

    to_be = ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'has',
             'have', 'had', 'do', 'did', 'does', 'can', 'could',
             'shall', 'should', 'will', 'would', 'may', 'might',
             'must', ]

    WH = ['WDT', 'WP', 'WP$', 'WRB', ]
    VB = ['VBG', 'VBD', 'VBN', 'VBP', 'VBZ', 'VB', ]
    VB_nogerund = ['VBD', 'VBN', 'VBP', 'VBZ', ]

    logic0 = numpy.in1d(tags[:-1, 1], ['IN']) * numpy.in1d(tags[1:, 1], WH)  # passive if true
    if numpy.any(logic0):
        return True

    logic1 = numpy.in1d(tags[:-2, 0], to_be) * numpy.in1d(tags[1:-1, 1], VB_nogerund) * numpy.in1d(tags[2:, 1],
                                                                                                   VB)  # chain of three verbs, active if true and previous not
    if numpy.any(logic1):
        return False

    if numpy.any(numpy.in1d(tags[:, 0], to_be)) * numpy.any(
            numpy.in1d(tags[:, 1], ['VBN'])):  # 'to be' + past participle verb
        return True

    # if no clauses have tripped thus far, it's probably active voice:
    return False

application = app = Flask(__name__)

@app.route('/',methods=["GET","POST"])
def upload():
    if request.method == 'POST':

        txt=request.form.get("text")

        results = nltk.regexp_tokenize(txt, pattern)
        results = [i for i in results if i not in pattern]
        results = [i for i in results if i not in stopworddic]
        results = [i.lower() for i in results]
        results = ' '.join(results)

        sumconc = 0
        sumword = 0
        for j, row in concdict.iterrows():
            k = len(re.findall(str(concdict.loc[j, "Word"]), results))
            sumword = sumword + k
            sumconc = sumconc + k * concdict.loc[j, "Conc"]
        conc=sumconc/sumword

        sumfreq = 0
        sumword = 0
        for j, row in freqdict.iterrows():
            k = len(re.findall(str(freqdict.loc[j, "word"]), results))
            sumword = sumword + k
            sumfreq = sumfreq + k * freqdict.loc[j, "freq"]
        freq=sumfreq/sumword

        results = nltk.regexp_tokenize(txt, pattern)
        results = [i for i in results if i not in pattern]
        num_word=len(results)
        num_page=num_word/844
        results = [i.lower() for i in results]
        results = ' '.join(results)

        eg = re.findall("for example", results)
        eg2 = re.findall('e\\.g\\.', txt)
        fi = re.findall("for instance", results)
        n = re.findall("namely", results)
        s = re.findall("such as", results)
        ai = re.findall(" as in ", results)
        examples = (len(eg) + len(fi) + len(n) + len(eg2) + len(s) + len(ai))/num_page

        fname=str(random.randint(0,999))
        my_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(my_path, fname+".txt")

        if num_word<2000:
            j=round(2000/num_word)+1
            txt_1=txt
            for k in range(0,j):
                txt=txt+"\n"+txt_1
            print(txt)

        with open(path, "w", encoding="UTF-8") as f:
            f.write(txt)
            sents=inaugural.sents(fileids=[path])

        j = 0
        k = 0
        for sent in sents:
            j = j + 1
            if is_passive_voice(str(sent)) == 0:
                k = k + 1

        if j>0:
            active=k/j

        if j==0:
            active=0

        os.remove(path)

        k=-1
        for j,row in percentile.iterrows():
            if conc>percentile.loc[j,"conc"]:
                k=j
        conc_per=k*5+5

        k=-1
        for j,row in percentile.iterrows():
            if examples>percentile.loc[j,"example"]:
                k=j
        example_per=k*5+5

        k=-1
        for j,row in percentile.iterrows():
            if freq>percentile.loc[j,"freq"]:
                k=j
        freq_per=k*5+5

        k=-1
        for j,row in percentile.iterrows():
            if active>percentile.loc[j,"active"]:
                k=j
        active_per=k*5+5

        clarity_per=round((conc_per+example_per+freq_per+active_per)/4)

        conc_color="yellow"
        if conc_per<40:
            conc_color="red"
        if conc_per>=70:
            conc_color="green"

        example_color="yellow"
        if example_per<40:
            example_color="red"
        if example_per>=70:
            example_color="green"

        freq_color="yellow"
        if freq_per<40:
            freq_color="red"
        if freq_per>=70:
            freq_color="green"

        active_color="yellow"
        if active_per<40:
            active_color="red"
        if active_per>=70:
            active_color="green"

        clarity_color="yellow"
        if clarity_per<40:
            clarity_color="red"
        if clarity_per>=70:
            clarity_color="green"

        k = -1
        for j, row in percentile_front.iterrows():
            if conc > percentile_front.loc[j, "conc"]:
                k = j
        conc_per_front = k * 5 + 5

        k = -1
        for j, row in percentile_front.iterrows():
            if examples > percentile_front.loc[j, "example"]:
                k = j
        example_per_front = k * 5 + 5

        k = -1
        for j, row in percentile_front.iterrows():
            if freq > percentile_front.loc[j, "freq"]:
                k = j
        freq_per_front = k * 5 + 5

        k = -1
        for j, row in percentile_front.iterrows():
            if active > percentile_front.loc[j, "active"]:
                k = j
        active_per_front = k * 5 + 5

        clarity_per_front=round((conc_per_front+example_per_front+freq_per_front+active_per_front)/4)

        conc_color_front = "yellow"
        if conc_per_front < 40:
            conc_color_front = "red"
        if conc_per_front >= 70:
            conc_color_front = "green"

        example_color_front = "yellow"
        if example_per_front < 40:
            example_color_front = "red"
        if example_per_front >= 70:
            example_color_front = "green"

        freq_color_front = "yellow"
        if freq_per_front < 40:
            freq_color_front = "red"
        if freq_per_front >= 70:
            freq_color_front = "green"

        active_color_front = "yellow"
        if active_per_front < 40:
            active_color_front = "red"
        if active_per_front >= 70:
            active_color_front = "green"

        clarity_color_front = "yellow"
        if clarity_per_front < 40:
            clarity_color_front = "red"
        if clarity_per_front >= 70:
            clarity_color_front = "green"

        score = {
            0 : str(conc)[0:6],
            1 : str(examples)[0:6],
            2 : str(freq)[0:6],
            3 : str(active)[0:6],
        }

        per = {
            0 : str(conc_per)+"%",
            1 : str(example_per)+"%",
            2 : str(freq_per)+"%",
            3 : str(active_per)+"%",
            4 : str(clarity_per)+"%",
        }

        col = {
            0 : str(conc_color),
            1 : str(example_color),
            2 : str(freq_color),
            3 : str(active_color),
            4 : str(clarity_color),
        }
        per_front = {
            0 : str(conc_per_front)+"%",
            1 : str(example_per_front)+"%",
            2 : str(freq_per_front)+"%",
            3 : str(active_per_front)+"%",
            4 : str(clarity_per_front)+"%",
        }

        col_front = {
            0 : str(conc_color_front),
            1 : str(example_color_front),
            2 : str(freq_color_front),
            3 : str(active_color_front),
            4 : str(clarity_color_front),
        }
        context = {
            'title': 'Article Writing Style Evaluation',
            'score': score,
            "percent":per,
            "color":col,
            "percent_front":per_front,
            "color_front":col_front,
        }
        return render_template('output.html',context=context)
    return render_template('upload.html')

if __name__=="__main__":
    # app.run(host='0.0.0,0',debug=True)
    # app.debug = True
    app.run()
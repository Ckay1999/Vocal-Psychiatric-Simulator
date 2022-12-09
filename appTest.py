from flask import Flask, render_template, request, Response
from modelSVM import SentimentAnalysisModel
from camera import VideoCamera

app = Flask(__name__)

model = SentimentAnalysisModel("svm_SK.sav", "train_dataset.csv")

Ques = ['How is your mood most of the time?',
        'Is there any fluctuation in the mood?',
        'How is your sleep? What is the pattern of sleep? Any difficulty in falling asleep or in getting up?',
        'Do you feel difficulty in concentration?',
        'Do you feel low or active most of the time?',
        'Do you feel uneasy or restless?',
        'How is your appetite? Do you have decreased or increased feeling of eating?',
        'How is your orientation towards sex? Interest in sex decreases?',
        'Are you losing interest in day to day activities?',
        'Do you see any variation in your weight?',
        'How would you rate this conversation? How helpful was it?']

Ans = []
Prediction = []

count = 0


@app.route('/')
def home():
    return render_template('try.html', question = Ques[count])

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/evaluation', methods=['GET', 'POST'])
def get_text():
	global count

	if request.method == "POST":
		count = count + 1
		textFromSpeech = request.form["textFromSpeech"]
		Ans.append(textFromSpeech)

		sentence = [textFromSpeech]
		sentence_vector = model.make_vector(sentence)
		Prediction.append(model.predict_sentiment(sentence_vector))

		if count == 11:
			return render_template('result.html', question = Ques, answer = Ans, prediction = Prediction, len = len(Ans))

	return render_template('try.html', question = Ques[count])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

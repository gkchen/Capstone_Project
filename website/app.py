from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        # Determine stock and columns to display
        keywords = request.form['keywords']
        if not keywords:
            keywords = 'data scientist'

        return render_template('index.html', id='application')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)

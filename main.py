from flask import Flask, request, render_template, send_file
import os
from __init__ import removeBg, unc

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # save the uploaded file
        f = request.files['image']
        filename = f.filename
        image_path = os.path.join(app.root_path, 'static', 'inputs', filename)
        f.save(image_path)
        # call the removeBg function
        removeBg(image_path)
        # get the output path
        output_filename = unc + '.png'
        output_path = os.path.join(app.root_path, 'static', 'results', output_filename)
        # render the template with output image and download button
        return render_template('index.html', output_path=output_path, output_filename=output_filename, input_file=filename)
    return render_template('index.html')


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    # construct the path to the file
    file_path = os.path.join(app.root_path, 'static', 'results', filename)
    # return the file to the user for download
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

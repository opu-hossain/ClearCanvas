from flask import Flask, request, render_template, send_file
import os
from __init__ import removeBg

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route for the index page.
    Handles both GET and POST requests.
    If the request method is POST, it saves the uploaded file,
    calls the removeBg function, gets the output path,
    and renders the index.html template with the output image and download button.
    If the request method is GET, it renders the index.html template.

    Returns:
        The rendered index.html template.
    """
    if request.method == 'POST':
        # Save the uploaded file
        # Get the uploaded file object
        f = request.files['image']
        # Get the filename of the uploaded file
        filename = f.filename
        # Construct the path to save the file
        image_path = os.path.join(app.root_path, 'static', 'inputs', filename)
        # Save the file
        f.save(image_path)

        # Call the removeBg function with the image path
        removeBg(image_path)

        # Get the output path
        output_filename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
        output_path = os.path.join(app.root_path, 'static', 'results', output_filename)

        # Render the index.html template with the output image and download button
        return render_template('index.html', output_path=output_path, output_filename=output_filename, input_file=filename)

    # If the request method is GET, render the index.html template
    return render_template('index.html')


@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    """
    Route for downloading the output image.

    Args:
        filename (str): Name of the file to be downloaded.

    Returns:
        The output image file for download.
    """

    # Construct the path to the file
    # This path includes the root path of the application,
    # the static folder, the results folder, and the filename.
    file_path = os.path.join(app.root_path, 'static', 'results', filename)
    
    # Return the file for download with the specified filename.
    # The `as_attachment=True` parameter sets the file to be downloaded
    # instead of displayed in the browser.
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

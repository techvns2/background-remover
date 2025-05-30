import os
from flask import Flask, request, send_file
from background_remover import remove_background

app = Flask(__name__)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    file = request.files['image']
    input_path = 'input.png'
    output_path = 'output.png'

    file.save(input_path)
    remove_background(input_path, output_path)
    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

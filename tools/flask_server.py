'''This is a flask http server for logger report'''

from flask import Flask, request
import time
import logging

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

file_handles = {}

@app.route('/')
def index():
    return {'msg': 'This is a simple http server for funcpipe debugging'}

@app.route('/logger', methods=['POST'])
def logger():
    log_t = time.time()
    msg = request.form['msg']
    mark = request.form['mark']
    log_msg = "{:.2f}   -   {}  -   {}\n".format(log_t, mark, msg)
    print(log_msg)
    write_file_log(mark, log_msg)
    return 'Hello World'

def write_file_log(mark, log_msg):
    if mark not in file_handles.keys():
        f = open(mark, "w")
        file_handles[mark] = f
    file_handles[mark].write(log_msg)
    file_handles[mark].flush()

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)


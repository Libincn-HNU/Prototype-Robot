# coding:utf-8

from flask import (Flask,render_template,url_for, jsonify, send_from_directory, request, flash)
import flask
from flask_cors import cross_origin, CORS
from pyltp import Parser
import datetime, time, json, codecs, os, xlrd
from werkzeug.utils import secure_filename

from synonyms_pos_syntax import *
from eda import gen_eda
from translate import translateExpand

LTP_DATA_DIR = '/export/resources/ltp_data_v3.4.0'   # LTP模型目录路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 分词模型路径， 模型名称为'parser.model'
parser = Parser()   # 初始化实例
parser.load(par_model_path)   # 加载模型

UPLOAD_FOLDER = 'demo' # 路径需要替换
ALLOWED_EXTENSIONS = set(['txt', 'xls', 'xlsx']) # 上传文件允许的后缀

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
app.secret_key='service in inter-credit'

@app.route('/')
def index():
    return("welcome enter data augement tools " + "*" * 100 )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/test/upload', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    print('upload  ' * 10)
    if request.method == 'POST':
        # file is null
        if 'file' not in request.files:
            flash('No file part')
            return response_data_decode({'status':1,'msg':''})
        file = request.files['file']
        print("*"*30)
        print(file)
        # file name is null
        if file.filename == '':
            flash('No selected file')
            return response_data_decode({'status':1,'msg':''})
        # success
        if file and allowed_file(file.filename):
            # save file begin
            test_mock_jss_save(file)
            time.sleep(1)
            # save file end
            return response_data_decode({'status': 0, 'msg': file.filename}) # 接收文件，返回文件路径
        # status  1-failed  0-success
    return response_data_decode({'status':1,'msg':''})
    

def deal(lines, checkValues):
    results = []
    for line in lines:
        if 'eda' in checkValues:
            print('eda')
            results.extend(gen_eda(line))
        if 'hmm' in checkValues:
            pass
        if 'synonyms' in checkValues:
            print('synonyms')
            tmp_results= deal_synonyms(line, parser)
            results.extend(tmp_results)
        if 'syntax' in checkValues:
            pass
        if 'translate' in checkValues:
            results.extend(translateExpand(line))

    parser.release()   # 释放模型
    return  results

@app.route('/test/run', methods=['GET', 'POST'])
@cross_origin()
def download_file():
    fileFlag = request.args.get("fileFlag")
    checkValues = request.args.get("checkValues")
    print("#" * 100 )
    print("fileFlag:" + str(fileFlag))
    print("checkValues:" + str(checkValues))
    print("#" * 100 )

    checkValues = checkValues.split(',')

    # run file and method begin
    suffix = fileFlag.split('.')[1] 
    print('*' * 100 )
    print(suffix)

    if suffix == 'txt':
        with open(os.path.join(app.config['UPLOAD_FOLDER'], fileFlag)) as f: 
            lines = f.readlines()

    elif suffix == 'xlsx':
        book = xlrd.open_workbook(os.path.join(app.config['UPLOAD_FOLDER'], fileFlag))
        sheet1 = book.sheet_by_name('Sheet1')
        lines = []
        for row in range(sheet1.nrows): # 遍历所有行
            lines.append( sheet1.cell(row, 0).value + '\t' + sheet1.cell(row, 1).value)

        lines = lines[1:]

    elif suffix == 'csv':
        pass
    else:
        print('*' * 100 )
        print("input file format error")

    results = deal(lines, checkValues)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'output.txt'), mode='w', encoding='utf-8') as f:
        f.writelines(results)

    time.sleep(5)
    # run file and method end

    results = [item.strip() + '\n' for item in results]


    return send_from_directory(app.config['UPLOAD_FOLDER'], 'output.txt', as_attachment=True)

@app.route('/test/template/download', methods=['GET'])
@cross_origin()
def download_template():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'demo/template.xlsx', as_attachment=True)

def test_mock_jss_save(file):
    # fileNameStr = file.filename.encode('utf-8')
    fileNameStr = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileNameStr))

def response_data_decode(data):
    return json.dumps(data)
    # return json.dumps(data).decode("unicode-escape")

def delete_file():
    # 文件定期删除,防止文件堆积
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5060, debug=True, threaded=True)

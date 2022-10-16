from flask import Flask, render_template, request, url_for , flash
import pandas as pd
import numpy as np
import os
from contextlib import redirect_stderr
import cv2
from flask import render_template , url_for , redirect , request , session
from werkzeug.utils import secure_filename
from twilio.rest import Client
import random
import smtplib
import bcrypt
from pymongo import MongoClient
import os 
import urllib.request
from tensorflow.keras.models import load_model
import urllib.request


connection_string="mongodb+srv://karthik:karthik@cluster0.rctxccl.mongodb.net/?retryWrites=true&w=majority"
client=MongoClient(connection_string)
dataSeesaws=client.Seesaws
collection=dataSeesaws.Users


app = Flask(__name__)
app.config['SECRET_KEY']="8aEaAjuuAXM8aC4"
IMAGE_SIZE = (150, 150)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

print("Loading Pre-trained Model ...")
model = load_model('model.h5')

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("Home.html")

@app.route("/login" , methods=['GET','POST'])
def login():
    emailid=request.form.get("email")
    password1=request.form.get("password")
    return render_template("Login.html")

@app.route("/register" , methods=['GET','POST'])
def register():
    if request.method == "POST":
        firstname=request.form.get("fname")
        lastname=request.form.get("lname")
        emailid=request.form.get("email")
        mobilenumber=request.form.get("phonenumber")
        password1=request.form.get("password")
        password2=request.form.get("confirmpassword")
        if password1 != password2:
                message = 'Passwords should match!'
                return render_template('register.html', message=message)
        else:
            hashed = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())
            user_input = {'firstName': firstname, 'lastName': lastname, 'email': emailid , 'Mobile': mobilenumber , 'Password': hashed}
            collection.insert_one(user_input)
            return render_template('VerifyMobile.html')
    return render_template("Register.html")

def generateotp():
    return random.randrange(100000,999999)

def getotpapi(number):
    account_sid="AC716e7536612de6dbc5a9f4105a82639d"
    app_token="9c58df4f77c9e8fc6cc844b4162fbce5"
    client=Client(account_sid,app_token)

    otp=generateotp()

    session['response']=str(otp)

    body='Your OTP is ' + str(otp)

    message=client.messages.create(
                            from_='+15736724358',
                            body=body,
                            to=number
    )

    if message.sid:
        return True 
    else:
        return False
    
@app.route("/verify/mobile",methods=["POST"])
def verify_mobile():
    if request.method == "POST":
        number=request.form.get("mnumber")
        print("number",number)
        number="+91"+str(number)
        val=getotpapi(number)
        if val:
            return render_template('Enterotp_number.html')
    return render_template("VerifyMobile.html")

@app.route('/verify/mobile/otp',methods=['POST'])
def validate():
    otp=request.form.get('otp')
    if 'response' in session:
        s=session['response']
        print(otp)
        print(session)
        session.pop('response',None)
        if s==otp:
            return render_template('VerifyEmail.html')
        else:
            return 'You are not apporized, Sorry!!'
    return render_template('Enterotp_number.html')


def getotpapi_email(emailid):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()

    s.login("seesawsofficial@gmail.com", "pinpekzdqvrcscqq")

    otp=generateotp()
    msg=otp + " is your OTP"

    s.sendmail('&&&&&&&&&&&&&&&&', emailid, msg)
    session['response']=str(otp)
    return True

@app.route("/verify/email",methods=['POST','GET'])
def verify_email():
    emailid = request.form.get('emailid')
    print(emailid)
    val=getotpapi_email(emailid)
    if val:
        return render_template('Enterotp_email.html')

@app.route('/verify/email/otp',methods=['POST'])
def validate_email():
    otp=request.form.get('otp')
    if 'response' in session:
        s=session['response']
        session.pop('response',None)
        if s==otp:
            return redirect("/login")
        else:
            return 'You are not apporized, Sorry!!'
    return render_template('Enterotp_email.html')


@app.route('/verifylogin',methods=['POST'])
def logged_in():
    if request.method == "POST":
        EmailId = request.form.get("email_address")
        Password = request.form.get("password")
        p1 = collection.find_one({"email": EmailId})
        if Password==p1["Password"]:
            return render_template('Home.html')
        else:
            return redirect(url_for("login"))


@app.route("/uploadimage")
def uploadfiles():
    return render_template('Upload.html')

@app.route("/uploadurl")
def uploadfile():
    return render_template('Uploadurl.html')

def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg/255.0
    currImg = np.reshape(currImg, (1, 150, 150, 3))
    return currImg


def model_pred(image):
    '''
    Perfroms predictions based on input image
    '''
    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    # Returns Probability:
    prediction = model.predict(image)[0]
    ans=np.argmax(prediction)
    # Returns class:
    # prediction = model.predict_classes(image)[0]
    print(prediction)
    return (ans)

@app.route("/predict",methods=['POST','GET'])
def predictimage():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)


        filelocation='static/uploads/'+filename
        print(filelocation)
        
        # Preprocessing Image
        image = image_preprocessor(filelocation)

        # Perfroming Prediction
        pred = model_pred(image)

        print(pred)
        return render_template('predict.html', name=filename, result=pred)

def download_image(url, file_path, file_name):
    full_path = file_path + file_name + '.jpg'
    urllib.request.urlretrieve(url, full_path)
    return full_path


@app.route("/predicturl",methods=['POST','GET'])
def predictimagebyurl():
    if request.method == "POST":
        url=request.form.get("fileurl")
        file_name = 'image1'
        d=download_image(url, 'static/uploads/', file_name)
        image = image_preprocessor(d)
        # Perfroming Prediction
        pred = model_pred(image)
        print(pred)
        return render_template('predict.html', name=file_name+'.jpg', result=pred)


if __name__ == "__main__":
    app.run(debug=True)
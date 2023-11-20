# Image Background Remover Web Application

## A fully functional Web Application written in Python that removes the Background of an Image Using AI.

This is a fun project of mine. In this project I have created a fully functional Web Application Using `flask` that removes the background of any image using AI Model. I have used Python for this project. Anyone can use this project for fun and learn something new.

NB:- This project is not for any commercial use.

## About This Project
My Goal was to reverse engineer the `https://www.remove.bg/` website. In My project , I Used an AI model "U2NET" that detect the Image subject and remove's the background. Note that this model is not perfect.
### WorkFlow
#### The project has 4 folders
1. model (It contains the AI that removes the background of the image )

2. static (It contains `CSS` for the style of web page, `inputs` directory for inputted image, `results` directory for Background removed image, and extra `masks`.)

3. templates (It contains the `index.html` file for the Web Interface)
4. saved_models (It contains Pre-Trained Model Data `u2net.pth`)

N:B Download the pre-trained `u2net.pth` model from `https://bit.ly/u2net` and paste it to `saved_models/u2net/` directory!. without the `u2net.pth` this application will not work.

#### The Project base has 3 file
1. config.py ( It configures the Model )
2. __init__.py ( It is the Main File the Removes the background of image )
3. main.py ( It connects the `__init__.py` with `flask` )


## Instructions

If you wish to run this project on your own device, make sure you have python installed in your device. then follow this instruction.

1. Make a new directory [ **I highly recommend using a virtual environment. Some modules have dependencies and it will prevent from clashes** ]
2. Open your terminal on that directory
3. Run `git clone https://github.com/Dark-D-E-V/Img-BG-Remover-Web.git`
4. After cloning Run `pip3 install -r requirements.txt`
5. Now just Run `python main.py`

### Don't know how to create a virtual environment? Follow this instruction:
1. Open your terminal
2. Run `pip3 install virtualenv` (If you don't have virtualenv installed. and you can check it by running `virtualenv --version`) 
3. Run `virtualenv venv`
4. Run `source venv/bin/activate`
5. Now you are in a virtual environment. Run `pip3 install -r requirements.txt`
6. Now Run `python main.py`

#### Or, you can follow this blog post: https://bit.ly/howtovenv

Now you are ready to Remove you're background:)!. For those who don't know "You'll see an IP with prot '5000'" `http://127.0.0.1:5000/` in your terminal. just open it to access the Website.

## Expectations for the contributor's

I have made the model as good as possible. I got help from other senior programmer's. But This model is still not have full accuracy. I expect from my fellow contributor's to update the model and make it as powerful as possible and make it a good AI project.

### For those who don't know Matching Learning

If You don't know matching learning for AI, it's fine. You can still contribute to the Web Interface. Make it as cool s your hart's desire :)...!
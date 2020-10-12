# Text-Analysis-Webapp
## Webapp for automating Text Analysis
![disp](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/textanalysis-app/display.png)<br><br>

<b> The purpose of this app is to offer anyone starting off an NLP projects a fast and convenient means of exploring the text data cutting down the time between EDA and Modelling<b><br>

To run the app:
1. Download google news 300 model and place it in `textanalysis-app/` directory.<br>Link: https://drive.google.com/file/d/1a_a-s_QvvYBJFCTKb2GBnJxJgPRBbJxR/view?usp=sharing
2. `pip install -r requirements.txt`
3. In the path `Text-Analysis-Webapp/textanalysis-app/` run the command `streamlit run app.py` <br>

The app will run in the local server.<br>
The app offers 4 main analysis options:<br>
1. Creating Word Cloud (using image mask as well)<br>
![img](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/images/snape-cloud.png)

2. N-gram Analysis<br>
![img](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/images/ng.png)

3. Part of Speech and NER Analysis<br>
![img](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/images/pos.png)     ![img](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/images/ner.png)

4. Similarity Analysis (training custom model or using pre-trained google news 300)<br>
![img](https://github.com/Arpan-Mishra/Text-Analysis-Webapp/blob/main/images/sa.png)


# Importing modules
import numpy as np
import streamlit as st
import pandas as pd
import cv2
import base64
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


df = pd.read_csv('music.csv')
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']


df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]


df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()

# Diving dataframe based on emotional & pleasant value in sorted order.
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]
def fun(list):
    data = pd.DataFrame()
    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':

            data = data.append(df_neutral.sample(n=t))

        elif v == 'Angry':

            data = data.append(df_angry.sample(n=t))

        elif v == 'fear':

            data = data.append(df_fear.sample(n=t))

        elif v == 'happy':

            data = data.append(df_happy.sample(n=t))
        else:

            data = data.append(df_sad.sample(n=t))

    elif len(list) == 2:
        # Row's count per emotion
        times = [20, 10]

        for i in range(len(list)):
            # Emotion name
            v = list[i]

            # Number of rows for this emotion
            t = times[i]

            if v == 'Neutral':
                # Adding rows to data
                data = data.append(df_neutral.sample(n=t))

            elif v == 'Angry':
                # Adding rows to data
                data = data.append(df_angry.sample(n=t))

            elif v == 'fear':
                # Adding rows to data
                data = data.append(df_fear.sample(n=t))

            elif v == 'happy':
                # Adding rows to data
                data = data.append(df_happy.sample(n=t))

            else:
                # Adding rows to data
                data = data.append(df_sad.sample(n=t))

    elif len(list) == 3:
        # Row's count per emotion
        times = [15, 10, 5]

        for i in range(len(list)):
            # Emotion name
            v = list[i]

            # Number of rows for this emotion
            t = times[i]

            if v == 'Neutral':
                # Adding rows to data
                data = data.append(df_neutral.sample(n=t))

            elif v == 'Angry':
                # Adding rows to data
                data = data.append(df_angry.sample(n=t))

            elif v == 'fear':
                # Adding rows to data
                data = data.append(df_fear.sample(n=t))

            elif v == 'happy':
                # Adding rows to data
                data = data.append(df_happy.sample(n=t))

            else:
                # Adding rows to data
                data = data.append(df_sad.sample(n=t))

    elif len(list) == 4:
        # Row's count per emotion
        times = [10, 9, 8, 3]

        for i in range(len(list)):
            # Emotion name
            v = list[i]

            # Number of rows for this emotion
            t = times[i]

            if v == 'Neutral':
                # Adding rows to data
                data = data.append(df_neutral.sample(n=t))

            elif v == 'Angry':
                # Adding rows to data
                data = data.append(df_angry.sample(n=t))

            elif v == 'fear':
                # Adding rows to data
                data = data.append(df_fear.sample(n=t))

            elif v == 'happy':
                # Adding rows to data
                data = data.append(df_happy.sample(n=t))

            else:
                # Adding rows to data
                data = data.append(df_sad.sample(n=t))
    else:
        # Row's count per emotion
        times = [10, 7, 6, 5, 2]

        for i in range(len(list)):
            # Emotion name
            v = list[i]

            # Number of rows for this emotion
            t = times[i]

            if v == 'Neutral':
                # Adding rows to data
                data = data.append(df_neutral.sample(n=t))

            elif v == 'Angry':
                # Adding rows to data
                data = data.append(df_angry.sample(n=t))

            elif v == 'fear':
                # Adding rows to data
                data = data.append(df_fear.sample(n=t))

            elif v == 'happy':
                # Adding rows to data
                data = data.append(df_happy.sample(n=t))

            else:
                # Adding rows to data
                data = data.append(df_sad.sample(n=t))
    return data




def pre(l):


    result = [item for items, c in Counter(l).most_common()
              for item in [items] * c]

    # Creating empty unique list
    ul = []

    for x in result:
        if x not in ul:
            ul.append(x)
    return ul



model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
# Softmax : It is mainly used to normalize neural networks output to fit between zero and one.
model.add(Dense(7, activation='softmax'))


model.load_weights('model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Not Attach OpenCL context to OpenCV.
cv2.ocl.setUseOpenCL(False)
# return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)

# Text or heading's

st.markdown("<h2 style='text-align: center; color: white;'><b>MUSIC MOOD LIFTER</b></h2>",
            unsafe_allow_html=True)
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('m9.jpg')

st.markdown(
    "<h5 style='text-align: center; color: black;'><b>Click on the name of recommended song to Play</b></h5>",
    unsafe_allow_html=True)

# Just for indentation
col1, col2, col3 = st.columns(3)

list = []
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION'):

        # Clearing values
        count = 0
        list.clear()

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count = count + 1

            for (x, y, w, h) in faces:
                # Creating rectangle around face
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                # Taking image out
                roi_gray = gray[y:y + h, x:x + w]
                # expand_dims() function is used to expand the shape of an array.
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)


                prediction = model.predict(cropped_img)


                max_index = int(np.argmax(prediction))
                list.append(emotion_dict[max_index])


                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

            # For emergency close window
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

            # To get time for window to stay, so that we take input
            if count >= 50:
                break


        cap.release()


        list = pre(list)

with col3:
    pass


fun(list)
new_df = fun(list)

# Just for separation
st.write("")

# Normal text
st.markdown("<h5 style='text-align: center; color: black;'><b>Recommended song's for you with artist names</b></h5>",
            unsafe_allow_html=True)
try:
    st.markdown("<h5 style='text-align: center; color: black;'><b>Your Mood Is </b><i>{}</i></h5>".format(list[0]), unsafe_allow_html=True)

except:
    pass
# Just for separation
st.write(
    "******************************************************************************************************")

try:
    # l = iterator over link column in dataframe
    # a = iterator over artist column in dataframe
    # i = iterator from (0 to 30)
    # n = iterator over name column in dataframe
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        # Recommended song name
        st.markdown("""<h4 style='text-align: center;'><a href={}>{} - {}</a></h4>""".format(l, i + 1, n),
                    unsafe_allow_html=True)

        # Artist name
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a), unsafe_allow_html=True)

        # Just for separation
        st.write(
            "***************************************************************************************************")
except:
    pass

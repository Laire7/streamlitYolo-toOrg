import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# upload시에 "upload"라는 폴더를 생성하고 그곳에 파일을 보관
# 폴더 생성
# 파일 저장

HOME = os.getcwd()
UPLOAD_DIR = os.path.join(HOME, 'uploads')

#폴더가 없을시에 생성
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

uploaded_files = st.file_uploader(
    "이미지 파일 선택", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    # 업로드된 데이터를 image에 저장
    # read() 함수의 의미는 네트워크를 통해 데니터를 가져온다
    image = uploaded_file.read() # 변수에만 저장. 프로그램 종료 후 저장 된 파일은 날라간다.
    # pil_image = Image.open(uploaded_file).convert("RGB")
    # np_image = np.asarray(pil_image)
    # cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    st.image(image, caption=uploaded_file.name)
    
    # 파일 uploads폴더에 저장
    with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
        f.write(image)
        f.close()
        st.success(f'{uploaded_file.name} 저장 완료!')
import streamlit as st
import pandas as pd
import openai

# Sidebar สำหรับการเติม API key
st.sidebar.title("OpenAI API Key")
api_key = st.sidebar.text_input("กรุณากรอก API Key", type="password")

# ตั้งค่า API key
if api_key:
    openai.api_key = api_key

    # ส่วนหลักของแอปพลิเคชัน
    st.title("NLP Application with OpenAI")
    user_input = st.text_area("กรุณาป้อนประโยคภาษาไทย")

    if user_input:
        try:
            # แปลประโยคเป็นภาษาฝรั่งเศส
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=f"แปลประโยคต่อไปนี้เป็นภาษาฝรั่งเศส:\n{user_input}",
                max_tokens=1000
            )
            translated_text = response.choices[0].text.strip()
            st.write("ข้อความแปลเป็นภาษาฝรั่งเศส:")
            st.write(translated_text)

            # วิเคราะห์คำศัพท์ที่น่าสนใจ
            response_vocabulary = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=(
                    f"โปรดแปลคำศัพท์ต่อไปนี้จากภาษาฝรั่งเศสเป็นภาษาไทย พร้อมตัวอย่างการใช้ในประโยค:\n\n"
                    f"{translated_text}\n\n"
                    "โปรดตอบในรูปแบบ 'คำศัพท์: คำแปล - ตัวอย่างการใช้'"
                ),
                max_tokens=1000
            )
            vocabulary_text = response_vocabulary.choices[0].text.strip()

            # แปลงข้อมูลเป็น DataFrame
            vocabulary_lines = [line for line in vocabulary_text.split('\n') if line and ':' in line]
            vocabulary_data = []
            for line in vocabulary_lines:
                parts = line.split(':', 1)  # แยกเฉพาะคำแรก
                if len(parts) == 2:
                    word = parts[0].strip()
                    rest = parts[1].split('-', 1)  # แยกเฉพาะคำแรกที่เจอ
                    if len(rest) == 2:
                        translation = rest[0].strip()
                        example = rest[1].strip()
                        vocabulary_data.append([word, translation, example])

            if vocabulary_data:
                df_vocabulary = pd.DataFrame(vocabulary_data, columns=["คำศัพท์", "คำแปล", "ตัวอย่างการใช้"])

                st.write("ตารางคำศัพท์ที่น่าสนใจ:")
                st.dataframe(df_vocabulary)

                # ดาวน์โหลดตารางคำศัพท์
                csv = df_vocabulary.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="ดาวน์โหลดตารางคำศัพท์เป็น CSV",
                    data=csv,
                    file_name='vocabulary.csv',
                    mime='text/csv',
                )
            else:
                st.warning("ไม่พบคำศัพท์ที่สามารถวิเคราะห์ได้จากข้อความที่แปล")
        except openai.error.OpenAIError as e:
            st.error(f"เกิดข้อผิดพลาดในการติดต่อ OpenAI API: {e}")
else:
    st.warning("กรุณากรอก API Key เพื่อเริ่มการทำงาน")

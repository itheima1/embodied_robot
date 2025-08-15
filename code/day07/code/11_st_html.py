import streamlit as st
import pandas as pd

st.write('dict字典形式的静态表格')
st.table(data={
    'name': ['张三', '李四', '王五'],
    'age': [18, 20, 22],
    'gender': ['男', '女', '男']
})

st.write('pandas中dataframe形式的静态表格')

df = pd.DataFrame(
    {
        'name': ['张三', '李四', '王五'],
        'age': [18, 20, 22],
        'gender': ['男', '女', '男']
    }
)
st.table(df)
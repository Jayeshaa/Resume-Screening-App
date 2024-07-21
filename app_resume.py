import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

kn_pkl = pickle.load(open('kn.pkl', 'rb'))
tf_pkl = pickle.load(open('tf.pkl', 'rb'))


def clean_resume(resume_txt):
    
    preprocess_txt = re.sub('http\S+\s',' ',resume_txt)
    
    preprocess_txt = re.sub('@\S+',' ',preprocess_txt)
    
    preprocess_txt = re.sub('#\S+\s',' ',preprocess_txt)
    
    preprocess_txt = re.sub('RT|cc',' ',preprocess_txt)
    
    preprocess_txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',preprocess_txt)
    
    preprocess_txt = re.sub(r'[^\x00-\x7f]',' ',preprocess_txt)
    
    preprocess_txt = re.sub('\s+',' ',preprocess_txt)
    
    return preprocess_txt

def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload your resumer here",type=['txt','docx','pdf'])
    if upload_file is not None:
        try:
            resume_r = upload_file.read()
            resume_txt = resume_r.decode('utf-8')
        except UnicodeDecodeError:
            resume_txt = resume_r.decode('latin-1')

        resume_txt = clean_resume(resume_txt)
        input_features = tf_pkl.transform([resume_txt])
        result = kn_pkl.predict(input_features)[0]
        st.write(result)

        #Map category id to category name
        
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(result, "Unknown")

        st.write("Predicted Category:", category_name)
if __name__ == '__main__':
    main()